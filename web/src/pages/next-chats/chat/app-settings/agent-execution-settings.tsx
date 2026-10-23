'use client';

import { Collapse } from '@/components/collapse';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useFetchChat } from '@/hooks/use-chat-request';
import {
  useFetchHarnessAgent,
  useFetchHarnessSkills,
  useUpdateHarnessAgent,
} from '@/hooks/use-harness-agent-request';
import { IHarnessAgentDefinition } from '@/interfaces/database/chat';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { parseRounds } from './agent-rounds';

/**
 * Edits the agent that answers this thread.
 *
 * The chat prompt shapes the reply; the agent decides how the reply is
 * researched — its instructions and the limits one run works inside. Saving
 * stores a new version, so turns already answered keep the definition they ran
 * with and the next turn picks up this one.
 */
export function AgentExecutionSettings() {
  const { t } = useTranslation();
  const { data: chat } = useFetchChat();
  const agentId = chat?.agent_id;
  const { data: agent } = useFetchHarnessAgent(agentId);
  const { updateAgent, loading } = useUpdateHarnessAgent();
  const { skills: availableSkills } = useFetchHarnessSkills(!!agentId);

  const [prompt, setPrompt] = useState('');
  // A numeric Input reports a number here, a text one a string: the box keeps
  // whatever it is given and the value is normalized only when it is sent.
  const [maxRounds, setMaxRounds] = useState<string | number>('0');
  const [approval, setApproval] = useState<'auto' | 'ask'>('ask');
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  const [version, setVersion] = useState<number>();

  useEffect(() => {
    if (!agent) {
      return;
    }
    setPrompt(agent.definition?.prompt ?? '');
    setMaxRounds(String(agent.definition?.policy?.max_rounds ?? 0));
    setApproval(agent.definition?.policy?.approval ?? 'ask');
    setSelectedSkills(agent.definition?.skills ?? []);
    setVersion(agent.version);
  }, [agent]);

  const definition = useMemo<IHarnessAgentDefinition | undefined>(() => {
    if (!agent) {
      return undefined;
    }
    return {
      ...agent.definition,
      prompt,
      policy: {
        ...agent.definition.policy,
        max_rounds: parseRounds(maxRounds),
        approval,
      },
      skills: selectedSkills,
    };
  }, [agent, prompt, maxRounds, approval, selectedSkills]);

  // A plain chat runs no agent, so there is nothing to edit and no section to show.
  if (!agentId) {
    return null;
  }

  const onSave = async () => {
    if (!definition) {
      return;
    }
    const saved = await updateAgent({ agentId, definition });
    if (saved) {
      setVersion(saved.version);
    }
  };

  return (
    <Collapse title={t('chat.agentExecution')}>
      <div className="flex flex-col gap-4">
        <div className="flex items-start justify-between gap-4">
          <span className="text-xs text-text-secondary">
            {t('chat.agentExecutionTip')}
          </span>
          {version ? (
            <Badge variant="secondary">
              {t('chat.agentVersion', { version })}
            </Badge>
          ) : null}
        </div>

        <div className="flex flex-col gap-1">
          <span className="text-sm">{t('chat.agentPrompt')}</span>
          <Textarea
            value={prompt}
            rows={8}
            onChange={(event) => setPrompt(event.target.value)}
            data-testid="agent-prompt"
          />
          <span className="text-xs text-text-secondary">
            {t('chat.agentPromptTip')}
          </span>
        </div>

        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-col gap-1">
            <span className="text-sm">{t('chat.agentMaxRounds')}</span>
            <span className="text-xs text-text-secondary">
              {t('chat.agentMaxRoundsTip')}
            </span>
          </div>
          <Input
            type="number"
            min={0}
            className="w-24"
            value={maxRounds}
            onChange={(event) => setMaxRounds(event.target.value)}
            data-testid="agent-max-rounds"
          />
        </div>

        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-col gap-1">
            <span className="text-sm">{t('chat.agentApproval')}</span>
            <span className="text-xs text-text-secondary">
              {t('chat.agentApprovalTip')}
            </span>
          </div>
          <Select
            value={approval}
            onValueChange={(value: 'auto' | 'ask') => setApproval(value)}
          >
            <SelectTrigger className="w-40" data-testid="agent-approval">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ask">{t('chat.agentApprovalAsk')}</SelectItem>
              <SelectItem value="auto">
                {t('chat.agentApprovalAuto')}
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex flex-col gap-1">
            <span className="text-sm">{t('chat.agentSkills')}</span>
            <span className="text-xs text-text-secondary">
              {t('chat.agentSkillsTip')}
            </span>
          </div>
          {availableSkills.length === 0 ? (
            <span className="text-xs text-text-secondary">
              {t('chat.agentSkillsEmpty')}
            </span>
          ) : (
            <div className="flex max-h-48 flex-col gap-2 overflow-y-auto pr-1">
              {availableSkills.map((skill) => (
                <label
                  key={skill.name}
                  className="flex items-start gap-2"
                  data-testid={`agent-skill-${skill.name}`}
                >
                  <Checkbox
                    className="mt-0.5"
                    checked={selectedSkills.includes(skill.name)}
                    onCheckedChange={(checked) =>
                      setSelectedSkills((current) =>
                        checked
                          ? [...current, skill.name]
                          : current.filter((name) => name !== skill.name),
                      )
                    }
                  />
                  <span className="flex flex-col">
                    <span className="text-sm">{skill.name}</span>
                    <span className="text-xs text-text-secondary">
                      {skill.description}
                    </span>
                  </span>
                </label>
              ))}
            </div>
          )}
        </div>

        <div className="text-right">
          <Button
            variant="outline"
            onClick={onSave}
            disabled={loading || !definition}
            data-testid="agent-settings-save"
          >
            {t('common.save')}
          </Button>
        </div>
      </div>
    </Collapse>
  );
}
