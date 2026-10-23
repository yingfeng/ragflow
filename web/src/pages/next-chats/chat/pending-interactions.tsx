import { InteractionCard } from '@/components/message-item/agent-activity';
import { IHarnessInteractionRef, useListPendingInteractions } from '@/hooks/use-chat-request';
import { IThinkEvent } from '@/interfaces/database/chat';
import { useEffect } from 'react';

/** Poll interval while a question is waiting; the run is blocked until answered. */
const PollIntervalMs = 10_000;

function toThinkEvent(interaction: IHarnessInteractionRef): IThinkEvent {
  return {
    kind: 'interaction',
    status: 'pending',
    summary: interaction.question,
    reason: interaction.context,
    request_id: interaction.request_id,
  };
}

/**
 * Questions waiting in the backend for this tenant.
 *
 * The stream renders a card when the question is asked live; this strip is for
 * the reload case, where the page has lost the stream and the run stays blocked
 * on an answer nobody can see.
 */
export function PendingInteractionsBanner() {
  const { data, refetch } = useListPendingInteractions(true);

  useEffect(() => {
    if (!data.length) return;
    const timer = setInterval(refetch, PollIntervalMs);
    return () => clearInterval(timer);
  }, [data.length, refetch]);

  if (!data.length) {
    return null;
  }

  return (
    <div className="mt-3 flex flex-col gap-2" data-testid="pending-interactions">
      {data.map((interaction) => (
        <InteractionCard
          key={interaction.request_id}
          event={toThinkEvent(interaction)}
        />
      ))}
    </div>
  );
}
