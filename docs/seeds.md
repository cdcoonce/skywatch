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
| 131-136 | 20 | planets | **held-out, reserved for #43** | not yet scored |
| 141-144 | 20 | planet-free | **held-out, reserved for #43** | not yet scored |
| 151-156 | 20 | planets | **held-out, reserved for #44** | not yet scored |
| 160-199 | any | either | design range for #43 | |
| 201-206 | 20 | planets | **held-out, reserved for #41** | not yet scored |
| 210-217 | 20 | planets | development | #45 design (the shape test's threshold was set on these and the development seeds); used |
| 218-219 | 20 | planet-free | development | #45 design; used |
| 221-226 | 20 | planets | development | held-out for #45 (the shape test), scored once; used |
| 231-234 | 20 | planet-free | development | held-out for #45 (the shape test), scored once; used |

Next unused range: 240 and up.

Also read for #45 (the statistic of every box `single_transit` reported, to set the shape threshold; nothing scored beyond the table in its PR): the stored runs (seeds 7 and 42, 3 and 20 nights; planet-free 41-43 at 3 nights), 11-14, 21-26, 31-50, 61-66, 71-76, 91-100 (planets) and the planet-free 41-44 and 51-58 at 20 nights.

Caveat on record: seeds 21-26 were where PR #34's review found the problems that became #35 and #36, so they were held out from those changes' thresholds but not from the choice of problem.
