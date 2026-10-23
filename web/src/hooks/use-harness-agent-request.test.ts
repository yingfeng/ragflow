import {
  fetchHarnessAgentRequest,
  updateHarnessAgentRequest,
} from './use-harness-agent-request';

const mockFetch = jest.fn();
const mockUpdate = jest.fn();

jest.mock('@/services/next-chat-service', () => ({
  __esModule: true,
  default: {
    fetchHarnessAgent: (...args: unknown[]) => mockFetch(...args),
    updateHarnessAgent: (...args: unknown[]) => mockUpdate(...args),
  },
}));

jest.mock('@/utils/api', () => ({
  __esModule: true,
  default: {
    harnessAgent: (agentId: string) => `/api/v1/harness/agents/${agentId}`,
  },
}));

/**
 * The parameterized harness service takes a native axios config. Without the
 * flag the service wraps the config in `data`, so the request goes out as
 * `/harness/agents/[object Object]` and the server answers 200 with a not-found
 * body — the editor then shows an empty agent that cannot be saved.
 */
describe('harness agent requests', () => {
  beforeEach(() => {
    mockFetch.mockReset();
    mockUpdate.mockReset();
    mockFetch.mockResolvedValue({ data: { code: 0, data: {} } });
    mockUpdate.mockResolvedValue({ data: { code: 0, data: {} } });
  });

  it('fetches by the id in the url, not by the config object', async () => {
    await fetchHarnessAgentRequest('agent-1');

    expect(mockFetch).toHaveBeenCalledWith(
      { url: '/api/v1/harness/agents/agent-1' },
      true,
    );
  });

  it('sends the definition as the body', async () => {
    const definition = { name: 'n', prompt: 'p', policy: { max_rounds: 0 } };

    await updateHarnessAgentRequest('agent-1', definition);

    expect(mockUpdate).toHaveBeenCalledWith(
      { url: '/api/v1/harness/agents/agent-1', data: definition },
      true,
    );
  });
});
