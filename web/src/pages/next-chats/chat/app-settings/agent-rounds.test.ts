import { parseRounds } from './agent-rounds';

describe('parseRounds', () => {
  it.each([
    ['0', 0],
    ['', 0],
    ['   ', 0],
    ['12', 12],
    [' 16 ', 16],
    ['-3', 0],
    ['abc', 0],
    // A numeric Input hands over a number, not a string: reading it with
    // `value.trim()` threw "value.trim is not a function" inside the editor.
    [12, 12],
    [0, 0],
    [null, 0],
    [undefined, 0],
    [Number.NaN, 0],
  ])('reads %j as %d', (input, expected) => {
    expect(parseRounds(input)).toBe(expected);
  });
});
