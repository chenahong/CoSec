# CoSec Scenario Examples

This directory contains two canonical scenarios that illustrate the schema
used by the CoSec evaluation pipeline:

- `static_example.json` is a Static112 scenario with a fixed community state
  and a fake authorization attack delivered through a user message.
- `dynamic_example.json` is a Dyn96 scenario with a community merge followed
  by a multi requester capability composition attack.

The files retain the complete executable scenario structure, including agent
views, setup state, interaction turns, hidden evaluator data, and verification
rules. They are provided as format examples only and do not constitute the
full CoSec dataset.
