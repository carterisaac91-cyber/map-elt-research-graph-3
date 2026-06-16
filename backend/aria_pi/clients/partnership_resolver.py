"""Partnership resolver — verifiable UNC ↔ company/sector links only.

Every fact returned here is tied to a verbatim source: a PubMed PMID + paper
title, an exact SEC filing sentence, or an official unc.edu web mention. There
are NO AI-generated summaries — the frontend renders the raw quotes so a
clinical reviewer can validate each claim against its primary source.

The resolver fans out to the existing free, keyless clients (PubMed, SEC EDGAR)
plus the optional web-search client. Sector queries are capped: companies are
ranked by total verifiable mention count and only the Top 10 are returned, so
the frontend payload can never balloon.
"""

import re
from datetime import datetime
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

from aria_pi.clients.pubmed_client import PubMedClient
from aria_pi.clients.sec_edgar_client import SECEdgarClient
from aria_pi.clients.web_search_client import WebSearchClient
from aria_pi.clients.nih_reporter_client import NIHReporterClient, unc_pis_from_grants
from aria_pi.clients.clinicaltrials_client import ClinicalTrialsClient
from aria_pi.sectors import seeds_for
from aria_pi.utils.name_resolver import normalize_company_name
from aria_pi.clients.confirmed_partners_client import (
    resolve_confirmed_partner,
    EMPTY_CONFIRMED,
)
from aria_pi.clients.unc_offerings_client import (
    resolve_unc_offerings,
    EMPTY_OFFERINGS,
)

_COI_WINDOW_YEARS = 5  # COI disclosures must be within the last N years

# UNC mention patterns used for verbatim SEC extraction.
_UNC_PATTERNS = [
    r"University of North Carolina",
    r"UNC[-\s]Chapel Hill",
    r"UNC Health",
    r"\bUNC\b",
]
_UNC_RE = re.compile("|".join(_UNC_PATTERNS))
_SECTOR_FANOUT = 12   # how many seed companies to evaluate for a sector query
_SECTOR_TOP = 10      # CRITICAL LIMIT: only the top-N ranked companies returned


# takes: a block of filing text (str)
# does: splits it into sentences and keeps only those that name UNC verbatim,
#       trimming each to a readable length
# returns: a list of verbatim sentence strings (deduped, max 3)
def _extract_unc_sentences(text: str) -> list:
    if not text:
        return []
    flat = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", flat)
    out, seen = [], set()
    for s in sentences:
        if _UNC_RE.search(s):
            clean = s.strip()[:480]
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                out.append(clean)
        if len(out) >= 3:
            break
    return out


# takes: a company name (str), an SECEdgarClient
# does: locates the company's CIK, pulls its latest 10-K narrative, and extracts
#       the verbatim sentences that mention UNC, with the filing URL as source
# returns: a dict { quotes: [str], filing_url: str, cik: str }
def resolve_sec_verbatim(company_name: str, sec: SECEdgarClient) -> dict:
    try:
        cik = sec._find_cik(company_name)
        if not cik:
            return {"quotes": [], "filing_url": "", "cik": ""}
        text = sec.get_tenk_text(cik, {})
        quotes = _extract_unc_sentences(text)
        filing_url = sec._latest_tenk_url(cik) if quotes else ""
        return {"quotes": quotes, "filing_url": filing_url, "cik": cik}
    except Exception as e:
        print(f"SEC verbatim error for {company_name}: {e}")
        return {"quotes": [], "filing_url": "", "cik": ""}


# takes: a company name (str), a PubMedClient
# does: finds UNC ↔ company co-authored papers and tallies the most frequent
#       author names across them
# returns: a dict { count: int, top_authors: [str], papers: [paper dicts] }
def resolve_pubmed(company_name: str, pubmed: PubMedClient) -> dict:
    try:
        papers = pubmed.search_unc_with_company(company_name, max_results=8) or []
        tally = {}
        for p in papers:
            for a in p.get("authors") or []:
                tally[a] = tally.get(a, 0) + 1
        top_authors = [a for a, _ in sorted(tally.items(), key=lambda kv: -kv[1])[:5]]
        return {"count": len(papers), "top_authors": top_authors, "papers": papers}
    except Exception as e:
        print(f"PubMed resolve error for {company_name}: {e}")
        return {"count": 0, "top_authors": [], "papers": []}


# takes: a company name (str), a PubMedClient
# does: finds UNC-authored papers that DISCLOSE a financial relationship with the
#       company (consulting fees, equity, research funding) and keeps only those
#       published within the last 5 years — the window in which such conflicts
#       must be disclosed
# returns: a dict { count: int, papers: [paper dicts], window_years: int }
def resolve_coi(company_name: str, pubmed: PubMedClient) -> dict:
    try:
        papers = pubmed.search_coi_disclosures(company_name, max_results=6) or []
        cutoff = datetime.now().year - _COI_WINDOW_YEARS
        recent = []
        for p in papers:
            yr = p.get("year")
            try:
                if not yr or int(yr) >= cutoff:
                    recent.append(p)
            except ValueError:
                recent.append(p)
        return {"count": len(recent), "papers": recent, "window_years": _COI_WINDOW_YEARS}
    except Exception as e:
        print(f"COI resolve error for {company_name}: {e}")
        return {"count": 0, "papers": [], "window_years": _COI_WINDOW_YEARS}


# takes: a company name (str), a PubMedClient
# does: attributes co-authored papers to specific UNC schools/centers so the
#       relationship can be traced to a concrete unit (and its web page)
# returns: a list of { unit: str, count: int } sorted by count desc
def resolve_unc_units(company_name: str, pubmed: PubMedClient) -> list:
    try:
        hits = pubmed.search_by_unc_schools(company_name, max_per_school=3) or []
        tally = {}
        for h in hits:
            unit = h.get("unc_school")
            if unit:
                tally[unit] = tally.get(unit, 0) + 1
        return [{"unit": u, "count": c} for u, c in sorted(tally.items(), key=lambda kv: -kv[1])]
    except Exception as e:
        print(f"UNC units resolve error for {company_name}: {e}")
        return []


# takes: a company name (str) and an optional SEC CIK (str)
# does: builds direct external links to the primary sources a reviewer can open
#       (PubMed search, SEC EDGAR company page, and a unc.edu web search)
# returns: a dict of { pubmed, edgar, unc_web } URLs
def build_links(company_name: str, cik: str = "") -> dict:
    q = quote_plus(company_name)
    pubmed = (f"https://pubmed.ncbi.nlm.nih.gov/?term="
              f"{quote_plus(company_name + ' AND University of North Carolina[Affiliation]')}")
    edgar = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K"
             if cik else f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={q}&type=10-K")
    unc_web = f"https://www.google.com/search?q={quote_plus('site:unc.edu ' + chr(34) + company_name + chr(34))}"
    return {"pubmed": pubmed, "edgar": edgar, "unc_web": unc_web}


# takes: a company name (str), a WebSearchClient
# does: queries `site:unc.edu "<company>"` for official UNC web mentions
# returns: a list of { title, url } result dicts (empty without a search key)
def resolve_ecosystem(company_name: str, web: WebSearchClient) -> list:
    try:
        return web.search_unc_site_mentions(company_name)
    except Exception as e:
        print(f"Ecosystem resolve error for {company_name}: {e}")
        return []


# takes: a company name (str)
# does: fans out concurrently to PubMed, SEC, and web search, gathering only
#       verifiable, source-linked partnership evidence for that one company
# returns: the full company partnership record (clinical / financial / ecosystem)
# takes: a company name (str), a PubMedClient
# does: runs the three PubMed lookups (co-authored, COI, UNC units) SEQUENTIALLY
#       so the burst never trips NCBI's keyless 3-requests/sec rate limit
# returns: a dict { clinical, coi, unc_units }
def resolve_pubmed_bundle(company_name: str, pubmed: PubMedClient) -> dict:
    return {
        "clinical": resolve_pubmed(company_name, pubmed),
        "coi": resolve_coi(company_name, pubmed),
        "unc_units": resolve_unc_units(company_name, pubmed),
    }


# takes: a company name (str) and an NIHReporterClient
# does: fetches UNC-awarded NIH grants whose text mentions the company and the
#       named UNC PIs on those grants
# returns: a dict { grants: [grant dicts], pis: [PI dicts] } — safe-empty on error
def safe_nih(company_name: str, client: NIHReporterClient) -> dict:
    """Fetch NIH grants mentioning company at UNC. Returns safe empty on failure."""
    try:
        grants = client.unc_grants_mentioning(company_name, max_results=5) or []
        pis = unc_pis_from_grants(grants, limit=3)
        return {"grants": grants, "pis": pis}
    except Exception as e:
        print(f"NIH safe error for {company_name}: {e}")
        return {"grants": [], "pis": []}


# takes: a company name (str) and a ClinicalTrialsClient
# does: finds trials the company sponsors and keeps the subset where UNC is a
#       listed site/collaborator (unc_signal is the matched facility/collab name)
# returns: a dict { all_count: int, unc_trials: [trial dicts] } — safe-empty on error
def safe_trials(company_name: str, client: ClinicalTrialsClient) -> dict:
    """Fetch clinical trials where company sponsors and UNC is a site. Returns safe empty on failure."""
    try:
        all_trials = client.search_by_sponsor(company_name) or []
        unc_trials = [t for t in all_trials if t.get("unc_signal")]
        return {"all_count": len(all_trials), "unc_trials": unc_trials[:4]}
    except Exception as e:
        print(f"Trials safe error for {company_name}: {e}")
        return {"all_count": 0, "unc_trials": []}


# takes: the user's company query (str) and an optional resolved official name
#        (str) to feed the strict SEC/Web clients
# does: fans out concurrently — PubMed gets the ORIGINAL query (it tolerates
#       typos), while SEC EDGAR + web search get the resolved official name so a
#       small typo no longer returns nothing from the exact-match clients. NIH
#       RePORTER + ClinicalTrials.gov add grant/trial-level UNC signals.
# returns: the full company partnership record, including `resolved_name`
def resolve_company(company_name: str, sec_web_name: str = None) -> dict:
    target = sec_web_name or company_name
    confirmed = resolve_confirmed_partner(company_name)
    unc_offerings = resolve_unc_offerings(sector) if sector else EMPTY_OFFERINGS
    pubmed, sec, web = PubMedClient(), SECEdgarClient(), WebSearchClient()
    nih_client, trials_client = NIHReporterClient(), ClinicalTrialsClient()
    results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {
            ex.submit(resolve_pubmed_bundle, company_name, pubmed): "pubmed",
            ex.submit(resolve_sec_verbatim, target, sec): "financial",
            ex.submit(resolve_ecosystem, target, web): "ecosystem",
            ex.submit(safe_nih, company_name, nih_client): "nih_grants",
            ex.submit(safe_trials, target, trials_client): "trials",
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    bundle = results.get("pubmed") or {}
    results["clinical"] = bundle.get("clinical")
    results["coi"] = bundle.get("coi")
    results["unc_units"] = bundle.get("unc_units")
    clinical = results.get("clinical") or {"count": 0, "top_authors": [], "papers": []}
    coi = results.get("coi") or {"count": 0, "papers": [], "window_years": _COI_WINDOW_YEARS}
    unc_units = results.get("unc_units") or []
    financial = results.get("financial") or {"quotes": [], "filing_url": "", "cik": ""}
    ecosystem = results.get("ecosystem") or []
    nih = results.get("nih_grants") or {"grants": [], "pis": []}
    trials = results.get("trials") or {"all_count": 0, "unc_trials": []}
    return {
        "query": company_name,
        "resolved_name": target,
        "type": "company",
        "links": build_links(target, financial.get("cik", "")),
        "clinical": clinical,
        "coi": coi,
        "unc_units": unc_units,
        "financial": financial,
        "ecosystem": ecosystem,
        "nih_grants": nih["grants"],
        "nih_pis": nih["pis"],
        "trials": trials["unc_trials"],
        "trials_total": trials["all_count"],
        "confirmed_interactions": confirmed,
        "unc_offerings": unc_offerings,
        "mention_count": clinical["count"] + coi["count"] + len(financial["quotes"]) + len(ecosystem),
    }


# takes: a sector name (str)
# does: evaluates the sector's seed companies for UNC partnership evidence,
#       ranks them by total verifiable mention count, and keeps only the Top 10
#       (CRITICAL LIMIT) so the payload stays small; aggregates their evidence
#       into the same three buckets the company view uses
# returns: the sector partnership record (ranked companies + aggregated buckets)
def resolve_sector(sector: str) -> dict:
    seeds = (seeds_for(sector) or [])[:_SECTOR_FANOUT]
    records = []
    # Keep concurrency low: each company already runs its PubMed lookups
    # sequentially, so too many parallel companies would still trip NCBI's rate
    # limit and zero out results.
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(resolve_company, name): name for name in seeds}
        for fut in as_completed(futures):
            try:
                records.append(fut.result())
            except Exception as e:
                print(f"Sector company error: {e}")
    records.sort(key=lambda r: r["mention_count"], reverse=True)
    top = records[:_SECTOR_TOP]

    clinical_papers, top_authors, financial_quotes, ecosystem, coi_papers = [], [], [], [], []
    unit_tally = {}
    for r in top:
        for p in (r["clinical"]["papers"] or [])[:2]:
            clinical_papers.append({**p, "company": r["query"]})
        top_authors.extend(r["clinical"]["top_authors"])
        for p in (r.get("coi", {}).get("papers") or [])[:2]:
            coi_papers.append({**p, "company": r["query"]})
        for u in (r.get("unc_units") or []):
            unit_tally[u["unit"]] = unit_tally.get(u["unit"], 0) + u["count"]
        for q in r["financial"]["quotes"]:
            financial_quotes.append({"company": r["query"], "text": q, "filing_url": r["financial"]["filing_url"]})
        for m in (r["ecosystem"] or [])[:2]:
            ecosystem.append({**m, "company": r["query"]})

    confirmed_in_sector = [
        r["confirmed_interactions"]
        for r in top
        if r.get("confirmed_interactions", {}).get("found")
    ]

    return {
        "query": sector,
        "resolved_name": sector,
        "type": "sector",
        "links": build_links(sector),
        "companies": [{"name": r["query"], "mention_count": r["mention_count"]} for r in top],
        "clinical": {
            "count": sum(r["clinical"]["count"] for r in top),
            "top_authors": list(dict.fromkeys(top_authors))[:8],
            "papers": clinical_papers[:12],
        },
        "coi": {
            "count": sum(r.get("coi", {}).get("count", 0) for r in top),
            "papers": coi_papers[:10],
            "window_years": _COI_WINDOW_YEARS,
        },
        "unc_units": [{"unit": u, "count": c} for u, c in sorted(unit_tally.items(), key=lambda kv: -kv[1])],
        "financial": {"quotes": financial_quotes[:10], "filing_url": ""},
        "ecosystem": ecosystem[:10],
        "confirmed_interactions": {
            "found": len(confirmed_in_sector) > 0,
            "confirmed_companies": [
                {"partner": c["partner"], "engagement_type": c["engagement_type"], "unc_unit": c["unc_unit"]}
                for c in confirmed_in_sector
            ],
        },
        "unc_offerings": resolve_unc_offerings(sector),
    }


# takes: a query string and a type ("company" | "sector")
# does: dispatches to the company or sector resolver
# returns: the resolver record dict for that query
def resolve_partnerships(query: str, type: str) -> dict:
    query = (query or "").strip()
    if not query:
        return {"query": "", "resolved_name": "", "type": type, "links": build_links(""),
                "clinical": {"count": 0, "top_authors": [], "papers": []},
                "coi": {"count": 0, "papers": [], "window_years": _COI_WINDOW_YEARS},
                "unc_units": [], "financial": {"quotes": [], "filing_url": ""}, "ecosystem": [],
                "confirmed_interactions": EMPTY_CONFIRMED}
    if type == "sector":
        return resolve_sector(query)
    # Correct typos before the strict SEC/Web clients see the name; PubMed still
    # receives the original query inside resolve_company.
    resolved = normalize_company_name(query)
    return resolve_company(query, sec_web_name=resolved)
