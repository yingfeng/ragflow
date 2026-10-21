import { useSetModalState } from '@/hooks/common-hooks';
import { useCreateChat } from '@/hooks/use-chat-request';
import { useFetchDefaultModelDictionary } from '@/hooks/use-llm-request';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AgentDraft, DefaultAgentDraft } from '../create-chat-dialog';

export const useCreateChatDialog = () => {
  const {
    visible: createChatVisible,
    hideModal: hideCreateChatModal,
    showModal: showCreateChatModal,
  } = useSetModalState();
  const { createChat, loading: createLoading } = useCreateChat();
  const { t } = useTranslation();
  const defaultModelDictionary =
    useFetchDefaultModelDictionary(createChatVisible);
  // The agent the new thread runs with. Disabled means a plain chat: the
  // request carries no `agent` at all, which is how the backend reads intent.
  const [agentDraft, setAgentDraft] = useState<AgentDraft>(DefaultAgentDraft);

  const InitialData = useMemo(
    () => ({
      name: '',
      icon: '',
      language: 'English',
      description: '',
      dataset_ids: [],
      prompt_config: {
        empty_response: '',
        prologue: t('chat.setAnOpenerInitial'),
        quote: true,
        keyword: false,
        tts: false,
        system: t('chat.systemInitialValue'),
        refine_multiturn: false,
        use_kg: false,
        reasoning: false,
        parameters: [
          { key: 'knowledge', optional: false },
          { key: 'date', optional: true },
        ],
        toc_enhance: false,
      },
      llm_id: defaultModelDictionary?.llm_id,
      tenant_llm_id: defaultModelDictionary?.llm_id,
      llm_setting: {},
      similarity_threshold: 0.2,
      keywords_similarity_weight: 0.7,
      top_n: 8,
      rerank_candidates_count: 64,
    }),
    [t, defaultModelDictionary?.llm_id],
  );

  const onCreateChatOk = useCallback(
    async (name: string) => {
      const ret = await createChat({
        ...InitialData,
        name,
        ...(agentDraft.enabled
          ? {
              agent: {
                policy: {
                  max_rounds: agentDraft.maxRounds,
                  approval: agentDraft.approval,
                },
              },
            }
          : {}),
      });
      if (ret === 0) {
        hideCreateChatModal();
        setAgentDraft(DefaultAgentDraft);
      }
    },
    [InitialData, createChat, hideCreateChatModal, agentDraft],
  );

  return {
    createChatLoading: createLoading,
    onCreateChatOk,
    createChatVisible,
    hideCreateChatModal,
    showCreateChatModal,
    agentDraft,
    setAgentDraft,
  };
};
