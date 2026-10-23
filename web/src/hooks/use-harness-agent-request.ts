import message from '@/components/ui/message';
import {
  IHarnessAgentDefinition,
  IHarnessAgentDetail,
} from '@/interfaces/database/chat';
import chatService from '@/services/next-chat-service';
import api from '@/utils/api';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

export enum HarnessAgentApiAction {
  FetchAgent = 'fetchHarnessAgent',
  UpdateAgent = 'updateHarnessAgent',
}

/**
 * Reads the definition the agent currently executes.
 *
 * The config is handed over as a native axios config (the `true` argument):
 * without it the service wraps it in `data`, the agent id reaches the server as
 * "[object Object]", and the answer is a 200 carrying a not-found body — which
 * looks like an agent with no definition instead of a broken request.
 */
export const fetchHarnessAgentRequest = (agentId: string) =>
  chatService.fetchHarnessAgent({ url: api.harnessAgent(agentId) }, true);

/** Replaces the definition, which the server stores as a new version. */
export const updateHarnessAgentRequest = (
  agentId: string,
  definition: IHarnessAgentDefinition,
) =>
  chatService.updateHarnessAgent(
    { url: api.harnessAgent(agentId), data: definition },
    true,
  );

/**
 * The definition the agent currently executes.
 *
 * It is the same version a run resolves, so what the editor shows is what the
 * agent does; a disabled id means the thread has no agent to edit.
 */
export const useFetchHarnessAgent = (agentId?: string) => {
  const { data, isFetching: loading } = useQuery<IHarnessAgentDetail | undefined>(
    {
      queryKey: [HarnessAgentApiAction.FetchAgent, agentId],
      enabled: !!agentId,
      gcTime: 0,
      refetchOnWindowFocus: false,
      queryFn: async () => {
        const { data } = await fetchHarnessAgentRequest(agentId!);
        return data?.code === 0
          ? (data.data as IHarnessAgentDetail)
          : undefined;
      },
    },
  );

  return { data, loading };
};

/**
 * Saves the edit as a new version.
 *
 * The answer carries the stored version, so the caller can show which version is
 * live; the thread picks it up on its next turn, without a restart.
 */
export const useUpdateHarnessAgent = () => {
  const queryClient = useQueryClient();

  const { isPending: loading, mutateAsync } = useMutation({
    mutationKey: [HarnessAgentApiAction.UpdateAgent],
    mutationFn: async ({
      agentId,
      definition,
    }: {
      agentId: string;
      definition: IHarnessAgentDefinition;
    }) => {
      const { data } = await updateHarnessAgentRequest(agentId, definition);
      if (data?.code === 0) {
        queryClient.invalidateQueries({
          queryKey: [HarnessAgentApiAction.FetchAgent, agentId],
        });
        message.success(data?.message || 'Success');
        return data.data as IHarnessAgentDetail;
      }
      message.error(data?.message ?? 'Unknown error');
      return undefined;
    },
  });

  return { loading, updateAgent: mutateAsync };
};
