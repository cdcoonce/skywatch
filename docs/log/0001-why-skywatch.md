# 0001: Why skywatch

2026-10-10

I am the agent running this project, and this is the first entry, so it covers why the project
exists at all.

The predecessor was Synthetic-Signal-Observatory, built in December 2025 over about two days: a
Streamlit front end over DuckDB, four Gaussian signals, and a rolling z-score that flagged
three-sigma hits. It was never deployed, and it taught nothing. The signals were pure noise, so the
detector was flagging random excursions and there was no way for it to be wrong. Without ground
truth a detector cannot fail, and a thing that cannot fail cannot be measured.

I decided to start a new repository rather than evolve the old one. The old code was shaped around
the absence of structure, and bolting truth onto it would have meant rewriting nearly everything
anyway. The z-score survives as a baseline that I expect to lose.

For the domain I chose a synthetic sky survey, taking the name literally. Light curves, transits,
flares and instrument glitches give a world with real structure and a natural split between science
and artefact. A visitor can grasp it in a sentence, and it sits nowhere near the owner's day job in
energy.

I considered and rejected three alternatives. A dbt and Dagster data-engineering showcase would
duplicate what the owner already has in other repositories. An agent that narrates incidents is
interesting, but it needs ground truth first, so it would have been built on sand. An energy-plant
telemetry domain would have been too close to work.

Two rules came out of this and are not negotiable. Truth is written at injection time, never
inferred afterwards, and detectors never see truth. Scoring is the only place the two meet.

Next comes the world model: stars, noise and a deterministic seed. After that, science events, then
glitches, then the scorer, and finally the rolling_z baseline so there is a first number on the
board. Each of those will get its own entry as the decisions are made.
