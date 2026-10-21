import { ApprovalCard } from '@/components/message-item/agent-activity';
import { useListPendingApprovals } from '@/hooks/use-chat-request';
import { IHarnessApproval, IThinkEvent } from '@/interfaces/database/chat';
import { useEffect } from 'react';

/** Poll interval while something is waiting; the run is blocked until answered. */
const PollIntervalMs = 10_000;

function toThinkEvent(approval: IHarnessApproval): IThinkEvent {
  return {
    kind: 'approval',
    tool: approval.tool,
    args: approval.args,
    status: 'pending',
    summary: `${approval.tool} · ${approval.risk}`,
    request_id: approval.request_id,
  };
}

/**
 * Approvals waiting in the backend for this tenant.
 *
 * The stream already renders a card for each request as it appears, so this only
 * matters after a reload: the page lost the stream, the run is still blocked, and
 * without this strip the only symptom would be a turn that never finishes.
 */
export function PendingApprovalsBanner() {
  const { data, refetch } = useListPendingApprovals(true);

  useEffect(() => {
    if (!data.length) return;
    const timer = setInterval(refetch, PollIntervalMs);
    return () => clearInterval(timer);
  }, [data.length, refetch]);

  if (!data.length) {
    return null;
  }

  return (
    <div className="mt-3 flex flex-col gap-2" data-testid="pending-approvals">
      {data.map((approval) => (
        <ApprovalCard
          key={approval.request_id}
          event={toThinkEvent(approval)}
        />
      ))}
    </div>
  );
}
