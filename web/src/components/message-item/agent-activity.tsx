/*
 *  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

import { Button } from '@/components/ui/button';
import { useResolveApproval, useResolveInteraction } from '@/hooks/use-chat-request';
import { IThinkEvent } from '@/interfaces/database/chat';
import {
  ChevronDown,
  ChevronRight,
  MessageCircleQuestion,
  ShieldQuestion,
  Sparkles,
  Wrench,
} from 'lucide-react';

/** A round the model spent working before it called a tool. */
const ReasoningKind = 'reasoning';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * One tool waiting for a human.
 *
 * The verdict may arrive from two places: the operator's own click (answered
 * here) and the stream, which replays it as a step with the same request id.
 * The event's own status wins once it arrives, so a second tab resolving the
 * same request does not leave a stale card behind.
 */
export function ApprovalCard({ event }: { event: IThinkEvent }) {
  const { t } = useTranslation();
  const { resolveApproval, loading } = useResolveApproval();
  const [localVerdict, setLocalVerdict] = useState<'granted' | 'denied'>();

  const status = event.status ?? 'pending';
  const verdict = status === 'pending' ? localVerdict : status;
  const canAnswer = status === 'pending' && !localVerdict && !!event.request_id;

  // remember persists a standing grant for this exact call (same tool, same
  // parameters), so the operator stops being asked the same question on every
  // round. Denying never remembers: a deny is about this call, not a rule.
  const handleDecide = useCallback(
    async (allow: boolean, remember = false) => {
      if (!event.request_id) return;
      const code = await resolveApproval({
        requestId: event.request_id,
        allow,
        remember,
      });
      if (code === 0) {
        setLocalVerdict(allow ? 'granted' : 'denied');
      }
    },
    [event.request_id, resolveApproval],
  );

  return (
    <div className="rounded-md border border-state-warning/40 bg-state-warning/5 p-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <ShieldQuestion className="size-4" />
        {t('chat.approvalTitle')}
      </div>
      <div className="mt-1 text-xs text-text-secondary">{event.summary}</div>
      {event.diff ? (
        <pre className="mt-2 max-h-64 overflow-auto rounded bg-bg-input p-2 text-xs whitespace-pre">
          {event.diff}
        </pre>
      ) : (
        event.args && (
          <pre className="mt-2 max-h-32 overflow-auto rounded bg-bg-input p-2 text-xs whitespace-pre-wrap break-all">
            {event.args}
          </pre>
        )
      )}
      <div className="mt-3 flex items-center gap-2">
        {canAnswer ? (
          <>
            <Button
              size="sm"
              variant="accent"
              disabled={loading}
              onClick={() => handleDecide(true)}
            >
              {t('chat.approvalAllow')}
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={loading}
              onClick={() => handleDecide(true, true)}
            >
              {t('chat.approvalAlways')}
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={loading}
              onClick={() => handleDecide(false)}
            >
              {t('chat.approvalDeny')}
            </Button>
          </>
        ) : (
          <span className="text-xs text-text-secondary">
            {verdict === 'granted'
              ? t('chat.approvalGranted')
              : verdict === 'denied'
                ? t('chat.approvalDenied')
                : t('chat.approvalPending')}
            {event.reason ? ` — ${event.reason}` : ''}
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * One question the run asked the user. Pending shows an input; answered shows
 * the pair. The answer posts by request id and the run resumes on the backend,
 * so the local echo is only there until the stream confirms it.
 */
export function InteractionCard({ event }: { event: IThinkEvent }) {
  const { t } = useTranslation();
  const { resolveInteraction, loading } = useResolveInteraction();
  const [draft, setDraft] = useState('');

  const status = event.status ?? 'pending';
  const canAnswer = status === 'pending' && !!event.request_id;

  const handleSend = useCallback(async () => {
    const answer = draft.trim();
    if (!event.request_id || !answer) return;
    const code = await resolveInteraction({
      requestId: event.request_id,
      answer,
    });
    if (code === 0) {
      setDraft('');
    }
  }, [draft, event.request_id, resolveInteraction]);

  return (
    <div className="rounded-md border border-state-info/40 bg-state-info/5 p-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <MessageCircleQuestion className="size-4" />
        {t('chat.interactionTitle')}
      </div>
      <div className="mt-1 text-xs text-text-secondary">{event.summary}</div>
      {canAnswer ? (
        <div className="mt-3 flex items-center gap-2">
          <input
            className="w-full rounded border border-border-button bg-bg-input px-2 py-1 text-xs"
            placeholder={t('chat.interactionPlaceholder')}
            value={draft}
            onChange={(change) => setDraft(change.target.value)}
            onKeyDown={(press) => {
              if (press.key === 'Enter' && !press.shiftKey) {
                press.preventDefault();
                void handleSend();
              }
            }}
          />
          <Button
            size="sm"
            variant="accent"
            disabled={loading || !draft.trim()}
            onClick={() => void handleSend()}
          >
            {t('chat.interactionSend')}
          </Button>
        </div>
      ) : (
        <div className="mt-2 text-xs text-text-secondary">
          {t('chat.interactionAnswered')}
        </div>
      )}
    </div>
  );
}

/**
 * An agent turn's activity: the approvals it is blocked on, the questions it
 * asked, the rounds it spent thinking, and the tool calls it made. Steps stay
 * collapsed by default — they are the working, not the answer — while an
 * approval or a question is always visible because the turn cannot finish
 * without it.
 *
 * The answer area holds only the turn's final text. A round that went on to call
 * a tool lands here instead, which is what keeps one round from overwriting the
 * text the next one prints.
 */
export function AgentActivity({ events }: { events?: IThinkEvent[] }) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  const [reasoningExpanded, setReasoningExpanded] = useState(false);

  const approvals = useMemo(
    () => (events ?? []).filter((event) => event.kind === 'approval'),
    [events],
  );
  const interactions = useMemo(
    () => (events ?? []).filter((event) => event.kind === 'interaction'),
    [events],
  );
  const reasoning = useMemo(
    () => (events ?? []).filter((event) => event.kind === ReasoningKind),
    [events],
  );
  // One usage step per turn: rounds, tokens, cost and prefix rebuilds. It is a
  // summary of the work, so it renders on its own rather than as another step.
  const usage = useMemo(
    () => (events ?? []).find((event) => event.kind === 'usage'),
    [events],
  );
  const steps = useMemo(
    () =>
      (events ?? []).filter(
        (event) =>
          event.kind !== 'approval' &&
          event.kind !== 'interaction' &&
          event.kind !== 'usage' &&
          event.kind !== ReasoningKind,
      ),
    [events],
  );

  if (!events?.length) {
    return null;
  }

  return (
    <div className="flex flex-col gap-2" data-testid="agent-activity">
      {approvals.map((event) => (
        <ApprovalCard key={event.request_id ?? event.summary} event={event} />
      ))}
      {interactions.map((event) => (
        <InteractionCard key={event.request_id ?? event.summary} event={event} />
      ))}
      {reasoning.length > 0 && (
        <div className="rounded-md border border-border-button">
          <button
            type="button"
            className="flex w-full items-center gap-2 px-3 py-2 text-xs text-text-secondary"
            onClick={() => setReasoningExpanded((value) => !value)}
            data-testid="agent-reasoning-toggle"
          >
            {reasoningExpanded ? (
              <ChevronDown className="size-3" />
            ) : (
              <ChevronRight className="size-3" />
            )}
            <Sparkles className="size-3" />
            {t('chat.agentReasoning', { count: reasoning.length })}
          </button>
          {reasoningExpanded && (
            <div className="flex flex-col gap-2 border-t border-border-button px-3 py-2">
              {reasoning.map((event, index) => (
                <p
                  key={`reasoning-${index}`}
                  className="whitespace-pre-wrap text-xs text-text-secondary"
                >
                  {event.summary}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
      {steps.length > 0 && (
        <div className="rounded-md border border-border-button">
          <button
            type="button"
            className="flex w-full items-center gap-2 px-3 py-2 text-xs text-text-secondary"
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? (
              <ChevronDown className="size-3" />
            ) : (
              <ChevronRight className="size-3" />
            )}
            <Wrench className="size-3" />
            {t('chat.agentSteps', { count: steps.length })}
          </button>
          {expanded && (
            <ul className="flex flex-col gap-1 border-t border-border-button px-3 py-2">
              {steps.map((step, index) => (
                <li
                  key={`${step.kind}-${step.tool ?? ''}-${index}`}
                  className="flex flex-wrap items-baseline gap-2 text-xs text-text-secondary"
                >
                  <span className="font-mono text-text-primary">
                    {step.tool ?? step.kind}
                  </span>
                  <span>{step.summary}</span>
                  {typeof step.duration_ms === 'number' && (
                    <span className="opacity-60">{step.duration_ms}ms</span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {usage && (
        <div
          className="text-xs text-text-secondary"
          data-testid="agent-usage"
          title={usage.summary}
        >
          {t('chat.agentUsage', {
            rounds: usage.rounds ?? 0,
            tokensIn: formatTokens(usage.tokens_in ?? 0),
            tokensOut: formatTokens(usage.tokens_out ?? 0),
          })}
          {usage.cost_usd
            ? t('chat.agentUsageCost', { cost: usage.cost_usd.toFixed(4) })
            : null}
          {usage.prefix_rebuilds
            ? t('chat.agentPrefixRebuilds', { count: usage.prefix_rebuilds })
            : null}
        </div>
      )}
    </div>
  );
}

/** Token counts are large; "34.5k" reads faster than "34512" in a step row. */
function formatTokens(count: number): string {
  if (count < 1000) return String(count);
  const thousands = count / 1000;
  return `${thousands >= 100 ? Math.round(thousands) : thousands.toFixed(1)}k`;
}
