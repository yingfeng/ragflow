import { MessageType } from '@/constants/chat';
import { IAnswer, IMessage } from '@/interfaces/database/chat';
import { buildMessageUuid } from '@/utils/chat';
import { omit } from 'lodash';
import { PrologueMessageIdPrefix } from './constants';

/**
 * Builds the assistant message that replaces the trailing placeholder.
 *
 * `fallbackId` carries the identity of the message being replaced. Stream
 * writes that arrive without a server id (a step event before the first text
 * chunk) would otherwise mint a fresh uuid, which changes the message's React
 * key and remounts it — collapsing every panel the user had expanded.
 */
export function buildAssistantMessageFromAnswer(
  answer: IAnswer,
  fallbackId?: string,
): IMessage {
  return {
    role: MessageType.Assistant,
    content: answer.answer,
    reference: answer.reference,
    id: buildMessageUuid({
      id: answer.id ?? fallbackId,
      role: MessageType.Assistant,
    }),
    prompt: answer.prompt,
    audio_binary: answer.audio_binary,
    ...omit(answer, 'reference'),
  } as IMessage;
}

/**
 * Builds the user question plus the empty assistant placeholder that the
 * streamed answer will progressively fill in.
 */
export function buildQuestionAndPlaceholder(message: IMessage): IMessage[] {
  return [
    {
      ...message,
      id: buildMessageUuid(message),
      awaitingServerId: true,
    },
    {
      role: MessageType.Assistant,
      content: '',
      conversationId: message.conversationId,
      id: buildMessageUuid({ ...message, role: MessageType.Assistant }),
    } as IMessage,
  ];
}

/**
 * Server messages don't carry the locally uploaded `File` instances. Re-attach
 * them by message id so attachments survive a server refresh.
 */
export function mergeLocalFiles(
  serverMessages: IMessage[],
  localMessages: IMessage[],
): IMessage[] {
  const filesMap = new Map(
    localMessages.filter((x) => x.files?.length).map((x) => [x.id, x.files]),
  );

  if (filesMap.size === 0) {
    return serverMessages;
  }

  return serverMessages.map((x) => ({
    ...x,
    files: filesMap.get(x.id) ?? x.files,
  }));
}

export function buildPrologueMessageId(conversationId: string) {
  return `${PrologueMessageIdPrefix}${conversationId}`;
}
