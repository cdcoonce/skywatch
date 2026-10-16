# Brainstorm: a real skywatch on TESS (2026-10-15)

**Problem.** skywatch's detectors have only ever seen a sky whose truth was written at injection time; the project should run them on a real sky (TESS light curves) and still be able to say, with numbers a visitor can check, how they did.

**Appetite.** A longer project: weeks of agent-driven sessions under the existing operating agreement (`personal/projects/skywatch.md` in the vault; summarised in the README), no hardware, no money. The first sector must publish a claim list within about three sessions, or the appetite is re-read.

**Direction.** Pre-registered claims, with an injected twin for calibration. For each TESS sector, the detectors run blind, the agent publishes its candidate list and the rule that produced it before any catalog is consulted, and then the catalogs score the list once, the way held-out seeds scored a PR. Because a catalog can confirm but rarely refute, every real run carries a twin: the same real light curves with the simulator's events injected, scored by the existing scorer. The twin gives recall and the false-alarm rate on that sky's own noise; the catalog gives the real finds. This bets that the project's honesty machinery is the product, and that a public list of claims checked afterwards is worth more to a visitor than a scoreboard alone.

**What "replace the synthetic world" means here.** The simulator stops generating stars, noise and glitches; real light curves supply those. It keeps generating events (transits, flares, pulsations, eclipses) for injection into the twin, with its current event mix. Refitting that mix to the catalogs is not planned; if a twin's result demands it, that is a new issue, measured like any other.

**Vocabulary.** A *field* is the set of stars observed together at one cadence, the set the field detector takes its median over. A *sector* is about 27 days of TESS data in two *orbits* of about 13.7 days with a gap between. A *claim* is a detection the agent publishes as a candidate. The *planet-free rule* in the repo (build logs 0008 and 0009) allows at most 5 detections on a 20-night run with no planets; its analogue here is a cap on claims per sector. A *bar* is an acceptance threshold written before the data is scored, stated in counts of objects with an absolute allowance (for example, "may lose at most one known planet"), per build log 0011.

**Decided in the brainstorm.**
- Nights become orbits. The two orbits of a sector are the detectors' "nights": fixed halves, not rolling. The first issue measures what a night-level statistic taken over 13.7 days does to each per-night detector. Changing a detector to make orbits meaningful is allowed, measured on the twin and on development sectors, before that detector's first claim list is registered. A detector that cannot be made to mean something on an orbit is retired on the record.
- Cadence: the first sector uses the pre-made 2-minute light curves. The field for the first sector is the 2-minute targets on one camera's CCD. Full-frame-image photometry (every star in a patch of sky) is a later issue, filed only if the field detector's measurement calls for it. Other cadences are later issues too.
- Nothing about a detector or a claim rule changes after that sector's claim list is registered. A sector is held out until its catalog is read; afterwards it is development data, as held-out seeds were. A sector's twin is development data from the start, so the first detector changes are measured on the first sector's twin.
- The claim rule's mechanism (what a claim is, with its kind taken from the detector that made it, and how the list is cut) is set in the PRD; the numeric cap for each sector is written in that sector's issue before the sector is downloaded. A sector that blows the cap still has its catalog read once and its verdicts recorded, then becomes development data.
- Catalog scoring happens once per sector and is recorded, with the date, in a sector ledger (`docs/sectors.md`, the seed ledger's analogue; `docs/seeds.md` stays for synthetic runs).
- Grades against the catalog: *confirmed* (a confirmed planet or a listed variable at that star with the claimed kind), *refuted* (the catalog records a different known cause at that star, such as an eclipsing binary claimed as a planet), *candidate* (a TESS Object of Interest, neither confirming nor refuting; its own column), *unmatched* (nothing known at that star; reported as a share, never as false alarms, because nobody has looked). Precedence when more than one applies: refuted, then confirmed, then candidate. The twin is scored by the existing scorer, where a false alarm is a detection on no injected event; the catalog grades are a new grader beside it.
- Storage: light curves are fetched by a repo command into a gitignored cache; scores, claim lists and catalog verdicts are small and are committed, so the page renders from committed files with no mission service in the render path.
- Agent-driven under the existing operating agreement; Charles reviews what ships.

**Criteria (frozen).**
1. Every published claim is graded confirmed, refuted, candidate or unmatched, and the unmatched share is reported.
2. The page keeps numbers a visitor can check; the scores, claim lists and verdicts behind them are committed, and the light curves are re-fetchable by a repo command.
3. Detectors run unchanged, or with changes an issue states and measures before the affected claim list is registered.
4. The page renders from committed files; no mission data service in the render path.
5. Appetite as above; the synthetic world's stars, noise and glitches are replaced, not calibrated.

**Killed options.**
- Fit the world to TESS, keep synthetic truth. Bet that calibration was the gap. Lost to criterion 5: a calibrated fake is still a fake, and the ask was a real sky.
- Catalog truth alone. Bet that known objects are enough truth. Lost because it is the chosen direction minus the pre-registration: the agent would see the catalog before deciding what to claim, which is the one thing the project's rules exist to prevent. Its grading survives inside the winner.
- Full-frame-image field as the direction. Bet that the field, not the star, is what skywatch is about. Lost because it writes no truth; it survives as a deferred ingest issue.
- Inject into real light curves as the direction. Bet that the truth mechanism is all that matters and the sky is a noise model. Lost on the frame: it never shows a real find, so the page would still read as synthetic. It survives as the twin.

**Premortem risks (winner).**
1. Unmatched claims dominate. Most TESS stars are unvetted, so the catalog neither confirms nor refutes most claims. Mitigation is the twin; the page leads with the twin's recall and false-alarm numbers, with the catalog verdicts beside them.
2. The detectors drown in red noise. Tuned on white synthetic noise, they may produce thousands of claims per sector. The cap and claim rule are written first, so this shows up as a blown cap on the record, not as a filter added after the fact.
3. Few known planets per field. Recall against the catalog is decided by a handful of objects, the trap of build log 0011. Bars on catalog recall are in objects with absolute allowances and are set per sector in its issue; recall is always reported with its count.
4. Orbits are not nights. A night-level statistic over 13.7 days may make a per-night detector meaningless rather than interesting; the first issue measures it and retirement is an allowed outcome.
5. Frame risk: a data-engineering project in disguise. Downloads, caching and ingest could eat weeks with no finding while the synthetic lab stalls. The three-session bound on the first claim list is the check.

**Open questions (deferred to the PRD).**
- Which sector and camera first.
- The numeric cap per sector and the exact claim rule (for example, top N by S/N per detector).
- Which catalogs, at which snapshot date, so a verdict is reproducible.
- How the twin chooses which stars get an injected event.

**Routing.** write-a-prd. This is feature work spanning ingest, scorer changes (four catalog grades), a claim-registration mechanism, a sector ledger and page changes; the PRD interview takes the open questions above, and prd-to-issues then produces the specced issues the build-review-land recipe expects.
