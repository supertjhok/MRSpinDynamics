# Writing Clear Documentation

This page is for maintainers adding or revising narrative documentation. The
site serves experimentalists, hardware designers, and software contributors,
so a page cannot assume that every reader already knows the package vocabulary
or the history of a feature.

## Begin with the reader's question

The opening should answer five questions before presenting equations, status
lists, or API names:

1. What physical or experimental problem does this page address?
2. Who should read it, and where should a newcomer start instead?
3. What result does the implemented model produce?
4. Which important approximation or boundary limits that result?
5. Which neighboring guide supplies the missing context?

Define a specialized term when it first changes how a result should be
interpreted. Do not begin with a phrase such as “measured particle
distribution,” “reference isochromat,” or “operating state” unless the page
first says what that object represents and whether it is measured, inferred,
or known only inside a simulation.

## Match the page to its role

| Page type | Lead with | Keep out of the opening |
|---|---|---|
| User guide | Goal, prerequisites, shortest runnable path | Historical milestone lists |
| Model reference | Physical mechanism, assumptions, outputs, validity range | Unexplained equations or class inventories |
| Engineering guide | Design decision, required inputs, realizability limits | Claims based only on illustrative hardware |
| Validation page | Claim under test, reference, tolerance, evidence level | A generic statement that tests pass |
| API reference | Signature and concise contract | Long tutorials that belong in a guide |
| Architecture record | Current status and link to the public guide | Language implying an implemented feature is still only planned |

Generated API and validation pages should be changed through their generators,
not edited by hand.

## Separate commands, states, and measurements

Many integrated simulations contain quantities that look similar but have
different evidentiary meaning. Name the distinction explicitly:

- a requested waveform is not necessarily the waveform delivered by hardware;
- a simulated ground-truth state is not a reconstructed measurement;
- a tissue image is not automatically a particle-concentration image;
- a fitted or illustrative parameter is not a calibrated hardware value; and
- a regression test is not experimental validation.

When a workflow crosses one of these boundaries, state which quantity the next
stage actually consumes.

## Present code after the model choice

Before a code example, explain why the selected workflow or model is appropriate
and identify the units of its important inputs. After the example, name the
outputs a reader should inspect and what would indicate failure. Long option
inventories belong in the generated API reference.

## Keep status and limitations current

Status blocks are useful after the opening context. Use “implemented,”
“validated,” and “calibrated” as different claims. When a feature moves from a
plan into the package, update its narrative guide, examples, known-gaps entry,
and any retained architecture record in the same change.

Before publishing, run the strict MkDocs build, check every new link, and inspect
the rendered desktop navigation and the pages whose openings or hierarchy
changed.
