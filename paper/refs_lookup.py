"""Look up real bibliographic metadata for the Related Work strands.

Why: the manuscript's Related Work must cite papers that exist and whose metadata we
actually read from the registry -- no citations from memory. Crossref is queried per
candidate title and the top hits are printed for a human pick; nothing is written yet.

ponytail: one flat query list + a title-similarity score. If a hit looks wrong, edit the
query string; no fuzzy framework needed.
"""
import json
import sys
import urllib.parse
import urllib.request

# (strand, intended key, search string)
QUERIES = [
    ("uda", "ganin2015dann", "Unsupervised domain adaptation by backpropagation"),
    ("uda", "ganin2016dann", "Domain-adversarial training of neural networks"),
    ("uda", "benjdira2019car", "Unsupervised domain adaptation for car detection in aerial images"),
    ("uda", "tuia2016survey", "Domain adaptation for the classification of remote sensing data an overview of recent advances"),
    ("uda", "zhu2022uda", "Deep subdomain adaptation network for image classification"),

    ("lshift", "lipton2018bbse", "Detecting and correcting for label shift with black box predictors"),
    ("lshift", "saerens2002adjusting", "Adjusting the outputs of a classifier to new a priori probabilities"),
    ("lshift", "garg2020unified", "A unified view of label shift estimation"),
    ("lshift", "azizzadenesheli2019regularized", "Regularized learning for domain adaptation under label shifts"),

    ("conformal", "tibshirani2019covariate", "Conformal prediction under covariate shift"),
    ("conformal", "gibbs2021aci", "Adaptive conformal inference under distribution shift"),
    ("conformal", "angelopoulos2024risk", "Conformal risk control"),
    ("conformal", "angelopoulos2023gentle", "A gentle introduction to conformal prediction and distribution-free uncertainty quantification"),

    ("sar", "huang2018opensarship", "OpenSARShip a dataset dedicated to Sentinel-1 ship interpretation"),
    ("sar", "xu2020fusar", "FUSAR-Ship building a high-resolution SAR-AIS matchup dataset of Gaofen-3 for ship detection and recognition"),
    ("sar", "wang2019ssdd", "A SAR dataset of ship detection for deep learning under complex backgrounds"),
    ("sar", "sun2019ais", "Automatic detection of ship targets in SAR images with AIS data"),
    ("sar", "chen2021shipcontext", "Ship classification in SAR imagery with context and auxiliary information"),

    ("selective", "geifman2017selective", "Selective classification for deep neural networks"),
    ("selective", "chow1970reject", "On optimum recognition error and reject tradeoff"),
    ("selective", "madras2018defer", "Predict responsibly improving fairness and accuracy by learning to defer to an expert"),
    ("selective", "mozannar2020consistent", "Consistent estimators for learning to defer to an expert"),

    ("negr", "bouthillier2021variance", "Accounting for variance in machine learning benchmarks"),
    ("negr", "musgrave2020reality", "A metric learning reality check"),
    ("negr", "sculley2018winners", "Winner's curse on pace progress and empirical rigor"),
    ("negr", "recht2019imagenet", "Do ImageNet classifiers generalize to ImageNet"),
]


def crossref(query, rows=3):
    url = ("https://api.crossref.org/works?rows=%d&select=DOI,title,author,container-title,"
           "issued,type,event&query.bibliographic=%s" % (rows, urllib.parse.quote(query)))
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # direct: registry is reachable
    req = urllib.request.Request(url, headers={"User-Agent": "manuscript-refs/1.0 (mailto:noreply@example.org)"})
    with op.open(req, timeout=30) as r:
        return json.load(r)["message"]["items"]


def show(item):
    au = item.get("author") or []
    names = "; ".join(("%s %s" % (a.get("given", ""), a.get("family", ""))).strip() for a in au[:4])
    venue = (item.get("container-title") or [""])[0] or (item.get("event") or {}).get("name", "")
    yr = (item.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return "%-58s | %s | %s | %s | %s" % (
        (item.get("title") or [""])[0][:58], item["DOI"], yr, venue[:34], names[:60])


def main():
    want = sys.argv[1:]  # optional: only these keys
    for strand, key, q in QUERIES:
        if want and key not in want:
            continue
        print("\n### %s / %s\n  q: %s" % (strand, key, q))
        try:
            for it in crossref(q):
                print("  " + show(it))
        except Exception as e:  # keep going; one dead lookup must not stop the sweep
            print("  LOOKUP FAILED: %r" % (e,))


main()
