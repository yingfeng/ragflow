import { MessageType } from '@/constants/chat';
import { useChatStreamStore } from './store';

const conversationId = 'conversation-1';

function startStream() {
  const store = useChatStreamStore.getState();
  store.ensureSession(conversationId, 'chat-1');
  store.appendQuestion(conversationId, {
    content: 'how many',
    id: 'question-1',
    role: MessageType.User,
    conversationId,
  });
  store.beginStream(conversationId, new AbortController());
}

function lastMessageId() {
  return useChatStreamStore.getState().sessions[conversationId].messages.at(-1)
    ?.id;
}

describe('applyAnswer', () => {
  beforeEach(() => {
    useChatStreamStore.setState({ sessions: {} });
  });

  // A step event can land before the first text chunk, so its write carries no
  // server id. Minting a new id per write changes the transcript's React key and
  // remounts the message, which collapses the Thinking / Agent steps panels the
  // user had expanded.
  it('keeps the streamed message identity when the write carries no id', () => {
    startStream();

    useChatStreamStore.getState().applyAnswer(conversationId, {
      answer: 'partial',
    });
    const first = lastMessageId();

    useChatStreamStore.getState().applyAnswer(conversationId, {
      answer: 'partial answer',
      think_events: [
        { kind: 'tool_call', tool: 'rag_search', summary: 'rag_search()' },
      ],
    });
    useChatStreamStore.getState().applyAnswer(conversationId, {
      answer: 'partial answer continued',
      think_events: [
        { kind: 'tool_call', tool: 'rag_search', summary: 'rag_search()' },
        { kind: 'tool_call', tool: 'rag_search', summary: 'rag_search()' },
      ],
    });

    expect(first).toBeTruthy();
    expect(lastMessageId()).toBe(first);
  });

  it('adopts the server id once the payload carries one', () => {
    startStream();

    useChatStreamStore.getState().applyAnswer(conversationId, {
      answer: 'partial',
    });
    useChatStreamStore.getState().applyAnswer(conversationId, {
      answer: 'partial answer',
      id: 'server-message-id',
    });

    expect(lastMessageId()).toBe('server-message-id');
  });

  it('keeps the adopted server id across later writes without one', () => {
    startStream();

    useChatStreamStore.getState().applyAnswer(conversationId, {
      answer: 'text',
      id: 'server-message-id',
    });
    useChatStreamStore.getState().applyAnswer(conversationId, {
      answer: 'text',
      think_events: [
        { kind: 'tool_call', tool: 'rag_search', summary: 'rag_search()' },
      ],
    });

    expect(lastMessageId()).toBe('server-message-id');
  });
});
