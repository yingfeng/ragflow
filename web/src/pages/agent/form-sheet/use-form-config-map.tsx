import { LlmSettingSchema } from '@/components/llm-setting-items/next';
import { CodeTemplateStrMap, ProgrammingLanguage } from '@/constants/agent';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';
import { Operator } from '../constant';
import AgentForm from '../form/agent-form';
import AkShareForm from '../form/akshare-form';
import AnswerForm from '../form/answer-form';
import ArXivForm from '../form/arxiv-form';
import BaiduFanyiForm from '../form/baidu-fanyi-form';
import BaiduForm from '../form/baidu-form';
import BeginForm from '../form/begin-form';
import BingForm from '../form/bing-form';
import CategorizeForm from '../form/categorize-form';
import CodeForm from '../form/code-form';
import CrawlerForm from '../form/crawler-form';
import DeepLForm from '../form/deepl-form';
import DuckDuckGoForm from '../form/duckduckgo-form';
import EmailForm from '../form/email-form';
import ExeSQLForm from '../form/exesql-form';
import GenerateForm from '../form/generate-form';
import GithubForm from '../form/github-form';
import GoogleForm from '../form/google-form';
import GoogleScholarForm from '../form/google-scholar-form';
import InvokeForm from '../form/invoke-form';
import IterationForm from '../form/iteration-form';
import IterationStartForm from '../form/iteration-start-from';
import Jin10Form from '../form/jin10-form';
import KeywordExtractForm from '../form/keyword-extract-form';
import MessageForm from '../form/message-form';
import PubMedForm from '../form/pubmed-form';
import QWeatherForm from '../form/qweather-form';
import RelevantForm from '../form/relevant-form';
import RetrievalForm from '../form/retrieval-form/next';
import RewriteQuestionForm from '../form/rewrite-question-form';
import { StringTransformForm } from '../form/string-transform-form';
import SwitchForm from '../form/switch-form';
import TavilyForm from '../form/tavily-form';
import TemplateForm from '../form/template-form';
import ToolForm from '../form/tool-form';
import TuShareForm from '../form/tushare-form';
import UserFillUpForm from '../form/user-fill-up-form';
import WenCaiForm from '../form/wencai-form';
import WikipediaForm from '../form/wikipedia-form';
import YahooFinanceForm from '../form/yahoo-finance-form';

export function useFormConfigMap() {
  const { t } = useTranslation();

  const FormConfigMap = {
    [Operator.Begin]: {
      component: BeginForm,
      defaultValues: {},
      schema: z.object({
        enablePrologue: z.boolean().optional(),
        prologue: z
          .string()
          .min(1, {
            message: t('common.namePlaceholder'),
          })
          .trim()
          .optional(),
        mode: z.string(),
        query: z
          .array(
            z.object({
              key: z.string(),
              type: z.string(),
              value: z.string(),
              optional: z.boolean(),
              name: z.string(),
              options: z.array(z.union([z.number(), z.string(), z.boolean()])),
            }),
          )
          .optional(),
      }),
    },
    [Operator.Retrieval]: {
      component: RetrievalForm,
      defaultValues: { query: [] },
      schema: z.object({
        name: z
          .string()
          .min(1, {
            message: t('common.namePlaceholder'),
          })
          .trim(),
        age: z
          .string()
          .min(1, {
            message: t('common.namePlaceholder'),
          })
          .trim(),
        query: z.array(
          z.object({
            type: z.string(),
          }),
        ),
      }),
    },
    [Operator.Generate]: {
      component: GenerateForm,
      defaultValues: {
        cite: true,
        prompt: t('flow.promptText'),
      },
      schema: z.object({
        prompt: z.string().min(1, {
          message: t('flow.promptMessage'),
        }),
      }),
    },
    [Operator.Answer]: {
      component: AnswerForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Categorize]: {
      component: CategorizeForm,
      defaultValues: {},
      schema: z.object({
        parameter: z.string().optional(),
        ...LlmSettingSchema,
        message_history_window_size: z.coerce.number(),
        items: z.array(
          z
            .object({
              name: z.string().min(1, t('flow.nameMessage')).trim(),
              description: z.string().optional(),
              // examples: z
              //   .array(
              //     z.object({
              //       value: z.string(),
              //     }),
              //   )
              //   .optional(),
            })
            .optional(),
        ),
      }),
    },
    [Operator.Message]: {
      component: MessageForm,
      defaultValues: {},
      schema: z.object({
        content: z
          .array(
            z.object({
              value: z.string(),
            }),
          )
          .optional(),
      }),
    },
    [Operator.Relevant]: {
      component: RelevantForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.RewriteQuestion]: {
      component: RewriteQuestionForm,
      defaultValues: {
        message_history_window_size: 6,
      },
      schema: z.object({
        llm_id: z.string(),
        message_history_window_size: z.number(),
        language: z.string(),
      }),
    },
    [Operator.Code]: {
      component: CodeForm,
      defaultValues: {
        lang: ProgrammingLanguage.Python,
        script: CodeTemplateStrMap[ProgrammingLanguage.Python],
        arguments: [],
      },
      schema: z.object({
        lang: z.string(),
        script: z.string(),
        arguments: z.array(
          z.object({ name: z.string(), component_id: z.string() }),
        ),
        return: z.union([
          z
            .array(z.object({ name: z.string(), component_id: z.string() }))
            .optional(),
          z.object({ name: z.string(), component_id: z.string() }),
        ]),
      }),
    },
    [Operator.WaitingDialogue]: {
      component: CodeForm,
      defaultValues: {},
      schema: z.object({
        arguments: z.array(
          z.object({ name: z.string(), component_id: z.string() }),
        ),
      }),
    },
    [Operator.Agent]: {
      component: AgentForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Baidu]: {
      component: BaiduForm,
      defaultValues: { top_n: 10 },
      schema: z.object({ top_n: z.number() }),
    },
    [Operator.DuckDuckGo]: {
      component: DuckDuckGoForm,
      defaultValues: {
        top_n: 10,
        channel: 'text',
      },
      schema: z.object({
        top_n: z.number(),
      }),
    },
    [Operator.KeywordExtract]: {
      component: KeywordExtractForm,
      defaultValues: { top_n: 3 },
      schema: z.object({
        llm_id: z.string(),
        top_n: z.number(),
      }),
    },
    [Operator.Wikipedia]: {
      component: WikipediaForm,
      defaultValues: {
        top_n: 10,
      },
      schema: z.object({
        language: z.string(),
        top_n: z.number(),
      }),
    },
    [Operator.PubMed]: {
      component: PubMedForm,
      defaultValues: { top_n: 10 },
      schema: z.object({
        top_n: z.number(),
        email: z.string(),
      }),
    },
    [Operator.ArXiv]: {
      component: ArXivForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Google]: {
      component: GoogleForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Bing]: {
      component: BingForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.GoogleScholar]: {
      component: GoogleScholarForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.DeepL]: {
      component: DeepLForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.GitHub]: {
      component: GithubForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.BaiduFanyi]: {
      component: BaiduFanyiForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.QWeather]: {
      component: QWeatherForm,
      defaultValues: {},
      schema: z.object({
        web_apikey: z.string(),
        lang: z.string(),
        type: z.string(),
        user_type: z.string(),
        time_period: z.string().optional(),
      }),
    },
    [Operator.ExeSQL]: {
      component: ExeSQLForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Switch]: {
      component: SwitchForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.WenCai]: {
      component: WenCaiForm,
      defaultValues: {
        top_n: 20,
      },
      schema: z.object({
        top_n: z.number(),
        query_type: z.string(),
      }),
    },
    [Operator.AkShare]: {
      component: AkShareForm,
      defaultValues: {
        top_n: 10,
      },
      schema: z.object({
        top_n: z.number(),
      }),
    },
    [Operator.YahooFinance]: {
      component: YahooFinanceForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Jin10]: {
      component: Jin10Form,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.TuShare]: {
      component: TuShareForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Crawler]: {
      component: CrawlerForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Invoke]: {
      component: InvokeForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Concentrator]: {
      component: () => <></>,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Note]: {
      component: () => <></>,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Template]: {
      component: TemplateForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Email]: {
      component: EmailForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Iteration]: {
      component: IterationForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.IterationStart]: {
      component: IterationStartForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.Tool]: {
      component: ToolForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.TavilySearch]: {
      component: TavilyForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.UserFillUp]: {
      component: UserFillUpForm,
      defaultValues: {},
      schema: z.object({}),
    },
    [Operator.StringTransform]: {
      component: StringTransformForm,
      defaultValues: {},
      schema: z.object({}),
    },
  };

  return FormConfigMap;
}
