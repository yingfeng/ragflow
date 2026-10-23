/**
 * Reads the round box as the number the run obeys.
 *
 * The argument is deliberately untyped: this app's `Input` rewrites the change
 * event of a numeric field so `event.target.value` is a `number`, while a plain
 * text field hands over a string. Both reach the form state, so both have to be
 * read here.
 *
 * Blank and 0 both mean "no cap": the agent definition stores 0 as absent, so an
 * empty box is how a user lifts the cap the chat was created with. Anything
 * unparseable is treated the same way rather than becoming a run that may not
 * take a single round.
 */
export function parseRounds(value: unknown): number {
  const parsed = Number.parseInt(String(value ?? '').trim(), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}
