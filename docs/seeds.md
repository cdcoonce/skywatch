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
| 151-156 | 20 | planets | **held-out, reserved for #44** | not yet scored |
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

Next unused range: 370 and up.

Also read for #45 (the statistic of every box `single_transit` reported, to set the shape threshold; nothing scored beyond the table in its PR): the stored runs (seeds 7 and 42, 3 and 20 nights; planet-free 41-43 at 3 nights), 11-14, 21-26, 31-50, 61-66, 71-76, 91-100 (planets) and the planet-free 41-44 and 51-58 at 20 nights.

Also read for #41 (the shape statistics of every star that reached the shape gate, binaries first, to design the depth-per-night reading and the chi-square bar; nothing scored beyond the table in its PR): the stored runs (seeds 7 and 42, 3 and 20 nights), 11-14, 21-26, 31-50, 61-66, 280-291 (20 nights), 280-299 (3 nights) and the planet-free 41-44 (20 nights) and 41-43 (3 nights).

Caveat on record: seeds 21-26 were where PR #34's review found the problems that became #35 and #36, so they were held out from those changes' thresholds but not from the choice of problem.
