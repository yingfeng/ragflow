import { ButtonLoading } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * What the create dialog collects about the agent that will run the thread.
 *
 * Empty `agent` in the request means a plain chat, so the switch is the whole
 * difference between "the pipeline answers" and "an agent answers".
 */
export type AgentDraft = {
  enabled: boolean;
  maxRounds: number;
  approval: 'auto' | 'ask';
};

export const DefaultAgentDraft: AgentDraft = {
  enabled: false,
  // Zero means "no round cap": the turn ends when the context window is spent
  // or the run stops making progress, which is how a long-horizon agent is
  // bounded. A positive value is an explicit tighter limit.
  maxRounds: 0,
  approval: 'ask',
};

type IProps = {
  hideModal: () => void;
  loading?: boolean;
  onOk: (name: string) => void;
  agent: AgentDraft;
  onAgentChange: (agent: AgentDraft) => void;
};

export function CreateChatDialog({
  hideModal,
  loading,
  onOk,
  agent,
  onAgentChange,
}: IProps) {
  const { t } = useTranslation();
  const [name, setName] = useState('');

  return (
    <Dialog open onOpenChange={hideModal}>
      <DialogContent
        className="sm:max-w-[480px]"
        data-testid="create-chat-modal"
      >
        <DialogHeader>
          <DialogTitle>{t('chat.createChat')}</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-5">
          <Input
            value={name}
            autoFocus
            onChange={(event) => setName(event.target.value)}
            placeholder={t('chat.createChatPlaceholder')}
            data-testid="create-chat-name"
          />

          <div className="flex items-start justify-between gap-4">
            <div className="flex flex-col gap-1">
              <span className="text-sm">{t('chat.agentEnabled')}</span>
              <span className="text-xs text-text-secondary">
                {t('chat.agentEnabledTip')}
              </span>
            </div>
            <Switch
              checked={agent.enabled}
              onCheckedChange={(enabled) =>
                onAgentChange({ ...agent, enabled })
              }
              data-testid="create-chat-agent-switch"
            />
          </div>

          {agent.enabled && (
            <div className="flex flex-col gap-4 rounded-md border border-border-button p-3">
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
                  value={agent.maxRounds}
                  onChange={(event) =>
                    onAgentChange({
                      ...agent,
                      maxRounds: Math.max(0, Number(event.target.value) || 0),
                    })
                  }
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
                  value={agent.approval}
                  onValueChange={(value: 'auto' | 'ask') =>
                    onAgentChange({ ...agent, approval: value })
                  }
                >
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ask">
                      {t('chat.agentApprovalAsk')}
                    </SelectItem>
                    <SelectItem value="auto">
                      {t('chat.agentApprovalAuto')}
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <ButtonLoading
            loading={loading}
            disabled={!name.trim()}
            onClick={() => onOk(name.trim())}
            data-testid="create-chat-save"
          >
            {t('common.save')}
          </ButtonLoading>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
