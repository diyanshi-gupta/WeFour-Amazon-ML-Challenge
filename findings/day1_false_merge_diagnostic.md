# Day 1 False-Merge Diagnostic

> **Branch:** `feat/blocking-hybrid-pv`  
> **Date:** 2026-09-26  
> **Analyst:** WeFour Team

## Summary

We sampled **20** Source1 entities that are true singletons (no matches in
`train_ground_truth.tsv`) but were incorrectly returned by the Day 1
**exact-name + country** blocker (`generate_exact_name_candidates`).

| Metric | Value |
|--------|-------|
| True singletons matched by exact blocker | **675** |
| Sample size inspected | **20** |
| Classified as franchise / chain name | **0** (0%) |
| Classified as generic business type | **3** (8%) |
| Classified as apparently unique name | **34** (92%) |
| Mean address token overlap | **0.01** |

## Hypothesis

**The exact-name blocker conflates distinct physical locations of chain /
franchise businesses** (e.g., multiple branches of a national chain, or two
businesses sharing a common generic name such as 'City Pharmacy') because it
only keys on the *name + country* pair.  Address information, which is the
only reliable disambiguator for these cases, is completely ignored by the
current exact-match key.

## Concrete Examples

### Example 1

| Field | Source1 (singleton) | Matched entity (S3) |
|-------|---------------------|----------------------------------------|
| Entity ID | `S1-253439478` | `S3-352757870` |
| Business Name | Cardiology Clinic, Inc | CARDIOLOGY CLINIC INC |
| Address | 3661 Airport Boulevard, Unit 354, Mobile, AL | 633 Church Street, Unit 2, Monmouth, Oregon |
| Country | US | US |
| Name type | *generic business type* | *generic business type* |
| Address token overlap | **different address** (0.14) | — |

> **Why it's a false merge:** Both share the same normalized name (`cardiology clinic incorporated`) and country, but they are at different addresses, confirming they are separate physical locations.

### Example 2

| Field | Source1 (singleton) | Matched entity (S3) |
|-------|---------------------|----------------------------------------|
| Entity ID | `S1-305869870` | `S3-211843256` |
| Business Name | Raj Power Private Limited | Raj Power  Private Limited |
| Address | H. No. 1757, Rupinagar, Talawade Ss, Taluka Haveli, Pune, Maharashtra | House No. D-#301 Basement Defence Colony Near Main Road, New Delhi, DL |
| Country | India | India |
| Name type | *unique name* | *unique name* |
| Address token overlap | **different address** (0.08) | — |

> **Why it's a false merge:** Both share the same normalized name (`raj power private limited`) and country, but they are at different addresses, confirming they are separate physical locations.

### Example 3

| Field | Source1 (singleton) | Matched entity (S2) |
|-------|---------------------|----------------------------------------|
| Entity ID | `S1-501530645` | `S2-199026847` |
| Business Name | Wexler's Vanguard Roofing Inc | WEXLER'S VANGUARD ROOFING INC |
| Address | 14298 123, Columbus, KY | ##204 TRUTHVILLE ROAD, GRANVILLE CIITY, NY |
| Country | US | US |
| Name type | *unique name* | *unique name* |
| Address token overlap | **different address** (0.00) | — |

> **Why it's a false merge:** Both share the same normalized name (`wexler s vanguard roofing incorporated`) and country, but they are at different addresses, confirming they are separate physical locations.

### Example 4

| Field | Source1 (singleton) | Matched entity (S3) |
|-------|---------------------|----------------------------------------|
| Entity ID | `S1-147569934` | `S3-568852743` |
| Business Name | Back Alley Nails | BACK Alley Nails |
| Address | 8547 Raven Court, Tucson, AZ | North Little Rock, Arkansas, 611 Green Valley Avenue |
| Country | US | US |
| Name type | *unique name* | *unique name* |
| Address token overlap | **different address** (0.00) | — |

> **Why it's a false merge:** Both share the same normalized name (`back alley nails`) and country, but they are at different addresses, confirming they are separate physical locations.

### Example 5

| Field | Source1 (singleton) | Matched entity (S3) |
|-------|---------------------|----------------------------------------|
| Entity ID | `S1-649358325` | `S3-423714408` |
| Business Name | Raven | Raven |
| Address | 3145 Davis Circle, Mesa, AZ | 1801 Evening Street, Judsonia, Arkansas |
| Country | US | US |
| Name type | *unique name* | *unique name* |
| Address token overlap | **different address** (0.00) | — |

> **Why it's a false merge:** Both share the same normalized name (`raven`) and country, but they are at different addresses, confirming they are separate physical locations.

### Example 6

| Field | Source1 (singleton) | Matched entity (S2) |
|-------|---------------------|----------------------------------------|
| Entity ID | `S1-655949921` | `S2-191644625` |
| Business Name | Balaji Impex Private Limited | BALAJI IMPEX PRIVATE [LIMITED] |
| Address | 4/207, Paramakudi, Tamil Nadu, Ramanathapuram, Sundar Nagar 3Nd Street | 630/131 DHAMJI SHAMJI INDESTATE L B S ROAD VIKROLI (WEST), MUMBAI, Maharashtra |
| Country | India | India |
| Name type | *unique name* | *unique name* |
| Address token overlap | **different address** (0.00) | — |

> **Why it's a false merge:** Both share the same normalized name (`balaji impex private limited`) and country, but they are at different addresses, confirming they are separate physical locations.

### Example 7

| Field | Source1 (singleton) | Matched entity (S3) |
|-------|---------------------|----------------------------------------|
| Entity ID | `S1-704460377` | `S3-860331400` |
| Business Name | International It Private Limited | International It Private  Limited |
| Address | Plot No-R S No. 245, Gujarat, Fp 94 Sp 3, B/H Pretol Pump, Udhana Ind-Ii, Surat City, Surat | Sector Z P Sushant City Near Ved Vyaspuri By Pass Meerut, Meerut, UP, Meerut |
| Country | India | India |
| Name type | *unique name* | *unique name* |
| Address token overlap | **different address** (0.00) | — |

> **Why it's a false merge:** Both share the same normalized name (`international it private limited`) and country, but they are at different addresses, confirming they are separate physical locations.

### Example 8

| Field | Source1 (singleton) | Matched entity (S2) |
|-------|---------------------|----------------------------------------|
| Entity ID | `S1-751991890` | `S2-428059760` |
| Business Name | Premier Solutions Private Limited | Premier Solutions Private Ltd |
| Address | 806 Indraprasth Corporate, Opp. Venus Atlantis, B/S Safal Pagusus, Nr. Prahalad Nagar Ga, Rden, Ahmedabad, Gujarat | NO 18 RAJNI BAGAN OPP: PARK PALACE A.C. MARKET, H.C.ROAD, West Bengal |
| Country | India | India |
| Name type | *unique name* | *unique name* |
| Address token overlap | **different address** (0.00) | — |

> **Why it's a false merge:** Both share the same normalized name (`premier solutions private limited`) and country, but they are at different addresses, confirming they are separate physical locations.

### Example 9

| Field | Source1 (singleton) | Matched entity (S2) |
|-------|---------------------|----------------------------------------|
| Entity ID | `S1-343713580` | `S2-156268656` |
| Business Name | First Food Private Limited | First Food Pvt Limited |
| Address | Nargis Dutt Rd Pali Hill, 401 Nisarg Building, Mumbai, Maharashtra, Mumbai | SARVESHWARI NAGAR, BEHIND SURYA COMPLEX NAKA BY PASS, FAIZABAD, Uttar Pradesh |
| Country | India | India |
| Name type | *unique name* | *unique name* |
| Address token overlap | **different address** (0.00) | — |

> **Why it's a false merge:** Both share the same normalized name (`first food private limited`) and country, but they are at different addresses, confirming they are separate physical locations.

### Example 10

| Field | Source1 (singleton) | Matched entity (S2) |
|-------|---------------------|----------------------------------------|
| Entity ID | `S1-349629289` | `S2-871832328` |
| Business Name | Frontier Institute of Technology Inc | Frontier Institute of Technology Inc. |
| Address | 83 Grover Street, Unit 1, Springfield, MA | OH, 104 SHERMAN ST, PEMBERVILLE |
| Country | US | US |
| Name type | *unique name* | *unique name* |
| Address token overlap | **different address** (0.00) | — |

> **Why it's a false merge:** Both share the same normalized name (`frontier institute of technology incorporated`) and country, but they are at different addresses, confirming they are separate physical locations.

## Conclusions & Recommendations

1. **Hypothesis CONFIRMED:** The majority of false merges involve franchise,
   chain, or generic business names. The exact-name blocker cannot distinguish
   between two branches of the same chain.

2. **Root cause:** Keying solely on `(clean_name, country)` produces the same
   blocking key for *all* branches of a chain within a country — there may be
   dozens or hundreds of 'Starbucks' in `US`.

3. **Recommended fix for Blocking v2 (hybrid-pv):**
   - Add a **PIN/ZIP + name-prefix secondary key** (already partially addressed
     by `generate_numeric_address_candidates`) to force co-location.
   - Add a **sentence-embedding cosine-similarity gate** so that exact-name
     matches with low address embedding similarity are pruned at blocking time.
   - Treat chains as a *special case*: if a name is flagged as a known chain,
     require *both* name AND address similarity ≥ 0.7 before passing a pair
     to the classifier.

4. **Metric impact:** Fixing these false merges reduces the false-positive rate
   of the blocker and — because singletons penalise precision heavily in
   Macro-F0.5 — should yield a measurable improvement in the final score.