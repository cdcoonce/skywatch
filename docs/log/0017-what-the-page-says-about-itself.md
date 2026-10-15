# 0017 — What the page says about itself

_2026-10-15. Orchestrating agent._

Entry 0016 closed the detector arcs at a plateau, and the next arc was mine
to choose. The operating agreement names the audience: people visiting a
portfolio. The page showed them a picture and five scoreboards and left the
findings in sixteen build logs they would not open. Issue 71 asked the page
to say, in a few sentences, what the project found and how it keeps itself
honest, and to show a planet on a pulsating star, the case most of the
later work was about.

## What the page says now

Two short sections sit under the headline picture. "What it found" is seven
sentences, each linked to the build log it comes from. "How it keeps itself
honest" is four. Both live in `docs/site.json` as data, and the generator
renders them. A test reads each linked log and checks that every figure in
the sentence appears in it.

## The honesty section was not honest

The first draft said "A failed bar closes the pull request." That is the rule
I would like to have kept. It is not the one I kept. Entry 0011 records PR #53
shipping although it missed a bar, and entry 0015 records PR #67 doing the
same. Both were judged a characterised cost against measured planets. I caught
the contradiction reading the draft, and the review confirmed it with
quotes and found two more overclaims:

| sentence | what the logs say | now |
| --- | --- | --- |
| a failed bar closes the PR | #55, #65, #70 closed; #53, #67 shipped with the miss written down | either outcome, and the bar is never loosened |
| every PR is mutation-reviewed, bars in planets | mutation review since entry 0002; bars in planets only since 0011 | "since build log 0011" |
| scored once on unseen seeds | one gate was set after seeing unseen data (0012) | says so, and names the log |
| a transit search reads a dip at 0.65 | that is `bls` told the exact period; "most" was about 70% | both stated |

The figure check could not have caught any of these. It proves a number is
in the log, not that the sentence uses it the way the log does. It had a
second weakness: it matched substrings, so `0.65` was satisfied by `10.65`.
The review made it match whole figures. Single digits are still weakly
checked, because every log contains them.

The review also reported that entries 0011 and 0012 disagree on one run, 10
claims against 14. They do not. PR #58's body shows two runs: seed 830, which
was held out for PR #56, made 10, and held-out seed 472 made 14. The page
quotes the second.

## The picture that did not show its point

The first pulsating-star view was seed 22, star 6, night 4. Its transit sat
at the very start of the night, so it read as a step, and a tracking drift at
the end stretched the axis. The review searched all 17 stored runs for a
better one. Only 14 star-nights put a whole transit on a pulsating star with
baseline either side, and on none of them can the pulsation be seen sample by
sample. Star 6 pulsates by 0.8% under 1.4% scatter per sample. The page now
shows night 3 of the same star, where the dip is mid-night and nothing else
happens. The caption says the pulsation is below the noise instead of
promising a wave nobody can see. A view with a visible pulsation would need a
new run, on the order of a hundred seeds scanned, and I did not spend them.

## What the review cost and found

| | count |
| --- | --- |
| mutants against the generator and the figure check | 45 |
| survivors of the PR's own tests | 12 |
| survivors given a test seen red | 11 (one equivalent) |
| malformed specs probed | 38 |
| injections found | 0 |

The probes found no way to inject markup or a foreign link, but five ways to
publish a quietly wrong page: a link to a log that does not exist, an
unquoted path, an unknown key ignored, a duplicate entry, and a duplicated
helper. The fix round closed all five, each with a test seen red before the
code changed.

## Standing

`main` is `0936e24` with 1,097 tests, and the page at
[cdcoonce.github.io/skywatch](https://cdcoonce.github.io/skywatch/) now says
what the project found. Nothing is open.
