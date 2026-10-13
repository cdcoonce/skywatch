# Seed ledger

Every survey seed a detector change has been designed on, checked against, or scored on, and what it is for. A **held-out** seed is scored once, after every threshold in a change is committed, and is never tuned on; once it has been scored it becomes development for every later change. Before naming held-out seeds in an issue, check this table and pick a range nobody has used. Add a row when you use a seed.

All runs are 200 stars. Planet-free runs (`pf-`) use the same seed with `planet_rate = 0`: the planets have their own random stream, so a planet-free run is the same field and the same glitches without the planets. A planet-free seed number and a planet seed number are separate uses.

| seeds | nights | kind | role now | history |
|---|---|---|---|---|
| 7, 42 | 3 and 20 | planets | development (the stored runs every log entry quotes) | since day one; 20-night runs since PR #29 |
| 11-14 | 20 | planets | development | PR #34 design |
| 21-26 | 20 | planets | development | held-out for #34, #37, #38 and #40; used |
| 31-50 | 20 | planets | development | PR #38, #40, #42 design |
| 41-44 | 20 | planet-free | development; the planet-free rule's reference runs | #33 threshold rule, #39 |
| 41-43 | 3 | planet-free | development | #33 |
| 51-58 | 20 | planet-free | development | PR #42 design |
| 61-66 | 20 | planets | development | never-seen check in PR #40's review; used |
| 71-76 | 20 | planets | development | held-out for #39 (PR #42); used |
| 81-84 | 20 | planet-free | development | held-out for #39 (PR #42); used |
| 91-100 | 20 | planets | development | PR #42's review; used |
| 101-104 | 20 | planet-free | development | PR #42's review; used |
| 111-114 | 20 | planets | development | PR #42's review (pulsator search); used |
| 131-136 | 20 | planets | development | held-out for #43 (PR #47), scored once; used |
| 141-144 | 20 | planet-free | development | held-out for #43 (PR #47), scored once; used |
| 151-156 | 20 | planets | development | held-out for #44 (the repeat test of a detrended box), scored once, `main` against the branch (transit true positives 117 to 116, pulsations reported as transits 1 to 0, nothing else changed); used |
| 160-199 | any | either | design range for #43 (not used by PR #47) | |
| 201-206 | 20 | planets | development | held-out for #41, scored twice (the `pick = 0` variant, reverted, then the final code; thresholds fixed in 9141d90 before both; no binary reached the shape gate on them); used |
| 210-217 | 20 | planets | development | #45 design (the shape test's threshold was set on these and the development seeds); used |
| 218-219 | 20 | planet-free | development | #45 design; used |
| 221-226 | 20 | planets | development | held-out for #45 (the shape test), scored once; used |
| 231-234 | 20 | planet-free | development | held-out for #45 (the shape test), scored once; used |
| 241-246 | 20 | planets | development | review of PR #46 (never-seen check of the shape test); used |
| 251-254 | 20 | planet-free | development | review of PR #46 (never-seen check of the shape test); used |
| 261-266 | 20 | planets | development | review of PR #47 (never-seen check of the flare-tail gate); used |
| 271-274 | 20 | planet-free | development | review of PR #47 (never-seen check of the flare-tail gate); used |
| 280-291 | 20 | planets | development | #41 design (the chi-square bar was set on these and the development seeds); used |
| 280-299 | 3 | planets | development | #41 design (the 3-night runs of the same seeds, 18 more binaries); used |
| 300-311 | 20 | planets | development | never-seen check of #41's change (builder), scored twice (the `pick = 0` variant and the final code); it decided the `pick = 0` quirk; used |
| 320-331 | 20 | planets | development | review of PR #49 (never-seen check of the depth-per-night reading and chi-square bar; no binary reached the shape gate); used |
| 340-369 | 3 | planets | development | review of PR #49 (never-seen check on 3-night runs, where binaries still leak; 15 binaries reached the shape gate, PR identical to `main`); used |
| 370-373 | 20 | planet-free | development | #44 design: every star sinusoidal (200 pulsators a run, `variability_weights={"sinusoidal": 1}`, written by `scripts/bls_detrend_audit.py --simulate`), so these are not the standard field of the same seed numbers; the repeat bar of 5 was read on these and the development runs; used |
| 374-377 | 20 | planets | development | #44 design: every star sinusoidal, `planet_rate` 0.5 (about 100 planets a run, a third of them on stars that go through the detrending); used |
| 380-391 | 20 | planets | development | review of PR #50 (never-seen check of the repeat test of a detrended box: one box refused, seed 390 star 136; `bls` and `single_transit` scored, `main` against the branch); used |
| 392-395 | 20 | planets | development | review of PR #50 (never-seen check; every star sinusoidal, `planet_rate` 0.5, written by `scripts/bls_detrend_audit.py --simulate`, so these are not the standard field of the same seed numbers; 4 boxes refused, 5 transit labels lost); used |
| 400-405 | 20 | planets | development | held-out for #48 (the decaying-run definition of a flare window), scored once, `main` against the branch, after the constants were committed (d3f46ce): transit true positives 32 to 32, false positives 3 to 3, eclipses and pulsations reported as transits 0 to 0, flares reported as transits 1 to 1, detections 37 to 37, per run 0, 4, 6, 0, 8, 19 on both; used |
| 410-413 | 20 | planet-free | development | held-out for #48, scored once with 400-405: 0 detections on each run, `main` and the branch; used |
| 420-425 | 20 | planets | development | held-out for #51 (`single_transit` on pulsating hosts), scored once, `main` against the branch (union transit labels bounded or touched 154 to 176 over all fourteen #51 runs, standard 36 to 39; fp 3 to 3; pulsation->transit 6 to 8); used |
| 426-429 | 20 | planets | development | held-out for #51 (every star sinusoidal, `planet_rate` 0.5, `scripts/bls_detrend_audit.py --simulate`, so not the standard field of the same seed numbers), scored once (union 118 to 137); used |
| 430-433 | 20 | planet-free | development | held-out for #51 (every star sinusoidal, `planet_rate` 0), scored once (no detection from `bls` or `single_transit`, before or after); used |
| 440-445 | 20 | planets | development | #48 design check (with the re-simulated seeds 12 and 261: `main` against the branch after the constants were fixed and before the held-out ranges, nothing changed afterwards: transit true positives 57 to 58, false positives 2 to 3); used |
| 446-449 | 20 | planet-free | development | #48 design check (0 detections on `main` and on the branch); used |
| 500-505 | 20 | planets | **reserved for PR #56's review** (standard field, never-seen check) | |
| 506-509 | 20 | planets | **reserved for PR #56's review** (every star sinusoidal, `planet_rate` 0.5) | |
| 510-513 | 20 | planet-free | **reserved for PR #56's review** (every star sinusoidal, `planet_rate` 0) | |
| 600-611 | 20 | planets | development | review of PR #52 (never-seen check of the decaying-run flare windows, `bls` scored, `main` against the branch: transit true positives 69 to 70, false positives 23 to 25, flares reported as transits 3 to 4, detections 107 to 112; seed 601 star 131 a false claim on a flare star, seed 609 star 150 a planet recovered); used |
| 620-623 | 20 | planet-free | development | review of PR #52 (never-seen check, `planet_rate` 0; 0 detections on each run, `main` and the branch); used |
| 700-711 | 20 | planets | development | review of PR #53 (never-seen check of `single_transit` on pulsating hosts, standard field; also `bls` with `trend_repeat_snr=5.0` for the #50 gate question); used |
| 712-715 | 20 | planets | development | review of PR #53 (never-seen check; every star sinusoidal, `planet_rate` 0.5, `scripts/bls_detrend_audit.py --simulate`, so not the standard field of the same seed numbers); used |
| 720-723 | 20 | planet-free | development | review of PR #53 (never-seen check; every star sinusoidal, `planet_rate` 0, `scripts/bls_detrend_audit.py --simulate`); used |
| 800-805 | 20 | planets | development | held-out for the #44 flip PR #55 (standard field), scored once; PR closed unmerged (decision rule missed by one planet); used |
| 806-809 | 20 | planets | development | held-out for PR #55 (every star sinusoidal, `planet_rate` 0.5), scored once; the lost planet is puls-seed809 star 185; used |
| 810-813 | 20 | planet-free | development | held-out for PR #55 (every star sinusoidal, `planet_rate` 0), scored once; used |
| 820-825 | 20 | planets | **held-out, reserved for #54** (standard field) | not yet scored |
| 826-829 | 20 | planets | **held-out, reserved for #54** (every star sinusoidal, `planet_rate` 0.5) | not yet scored |
| 830-833 | 20 | planet-free | **held-out, reserved for #54** (every star sinusoidal, `planet_rate` 0) | not yet scored |

Next unused range: 450-499 and 514-599 (free), and 840 and up. (440-449 are #52's design seeds; 600-623 and 700-723 are review seeds; 800-833 are used; 500-513 are reserved above.)

Also read for #48 (every flare-sized masked run of `bls`, 23,338 of them on 10,000 stars, with the truth under it, to read the shape of a flare, a hot pixel and a cosmic ray and to set the decay-time band and the residual bar; `scripts/bls_flare_run_audit.py`): seeds 7, 42, 21-26, 31-40, 61-66 and 71-76 with planets and planet-free 41-44, 51-58, 81-84 and 101-104 (all 20 nights, 200 stars). `main` against the branch was scored on all of these but 31-40 (40 runs), and on 12, 261, 440-445 and planet-free 446-449.

Also read for #44 (the 67 boxes accepted after a pulsation subtraction, to set the repeat bar; scored beyond the table in its PR only on 151-156): every planet and planet-free run of the ranges above that was simulated again for it (seeds 11-14, 31-50, 61-66, 71-76, 91-100, 111-114, 131-136, 201-206, 210-217, 221-226, 241-246, 261-266, 280-291, 300-311 and 320-331 with planets; 51-58, 81-84, 101-104, 141-144, 218-219, 231-234, 251-254 and 271-274 planet-free; all 20 nights, 200 stars) and the stored runs (seeds 7 and 42 at 3 and 20 nights, 21-26, the planet-free 41-44 and 41-43 at 3 nights).

Also read for #45 (the statistic of every box `single_transit` reported, to set the shape threshold; nothing scored beyond the table in its PR): the stored runs (seeds 7 and 42, 3 and 20 nights; planet-free 41-43 at 3 nights), 11-14, 21-26, 31-50, 61-66, 71-76, 91-100 (planets) and the planet-free 41-44 and 51-58 at 20 nights.

Also read for #41 (the shape statistics of every star that reached the shape gate, binaries first, to design the depth-per-night reading and the chi-square bar; nothing scored beyond the table in its PR): the stored runs (seeds 7 and 42, 3 and 20 nights), 11-14, 21-26, 31-50, 61-66, 280-291 (20 nights), 280-299 (3 nights) and the planet-free 41-44 (20 nights) and 41-43 (3 nights).

Also read for #51 (the host table of `scripts/single_transit_pulsator_audit.py` and the 18 labels; nothing scored beyond the table in its PR but the 51 runs where `single_transit` changes, `main` against the branch): every planet and planet-free run of the development ranges above that was simulated again for it, all 20 nights and 200 stars (149 standard planet runs: 7, 42, 11-14, 21-26, 31-50, 61-66, 71-76, 91-100, 111-114, 131-136, 151-156, 201-206, 210-217, 221-226, 241-246, 261-266, 280-291, 300-311, 320-331 and 380-391; 38 standard planet-free: 41-44, 51-58, 81-84, 101-104, 141-144, 218-219, 231-234, 251-254 and 271-274; the all-sinusoidal 374-377 and 392-395 with planets and 370-373 planet-free). No design seed from 440 up was used.

Caveat on record: seeds 21-26 were where PR #34's review found the problems that became #35 and #36, so they were held out from those changes' thresholds but not from the choice of problem.
