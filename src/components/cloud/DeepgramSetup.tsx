import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { openUrl } from "@tauri-apps/plugin-opener";
import { Check, ExternalLink, Loader2, X } from "lucide-react";
import { commands } from "@/bindings";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";

export const DEEPGRAM = "deepgram";
export const DEEPGRAM_DEFAULT_MODEL = "nova-3";

const SIGNUP_URL = "https://console.deepgram.com/signup";
const CONSOLE_URL = "https://console.deepgram.com/";
const KEY_DOCS_URL =
  "https://developers.deepgram.com/docs/create-additional-api-keys";

/** Deepgram models worth one click. nova-3 is the default. */
export const DEEPGRAM_MODELS: ReadonlyArray<{ id: string; hintKey: string }> = [
  { id: "nova-3", hintKey: "nova3" },
  { id: "flux-general-en", hintKey: "fluxEn" },
  { id: "flux-general-multi", hintKey: "fluxMulti" },
];

export type KeyCheckState =
  | { status: "idle" }
  | { status: "checking" }
  | { status: "valid" }
  | { status: "invalid"; message: string };

/** Check a key against Deepgram without sending audio. */
export const useDeepgramKeyCheck = () => {
  const [state, setState] = useState<KeyCheckState>({ status: "idle" });

  const check = async (apiKey: string): Promise<boolean> => {
    setState({ status: "checking" });
    const result = await commands.testCloudSttApiKey(DEEPGRAM, apiKey.trim());
    if (result.status === "ok") {
      setState({ status: "valid" });
      return true;
    }
    setState({ status: "invalid", message: result.error });
    return false;
  };

  const reset = () => setState({ status: "idle" });

  return { state, check, reset };
};

export const KeyCheckStatus: React.FC<{ state: KeyCheckState }> = ({
  state,
}) => {
  const { t } = useTranslation();
  switch (state.status) {
    case "checking":
      return (
        <p className="flex items-center gap-1.5 text-xs text-text/60">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
          {t("cloudSetup.key.checking")}
        </p>
      );
    case "valid":
      return (
        <p className="flex items-center gap-1.5 text-xs text-green-500">
          <Check className="w-3.5 h-3.5" />
          {t("cloudSetup.key.valid")}
        </p>
      );
    case "invalid":
      return (
        <p className="flex items-start gap-1.5 text-xs text-red-400">
          <X className="w-3.5 h-3.5 mt-px shrink-0" />
          {state.message}
        </p>
      );
    default:
      return null;
  }
};

const LinkButton: React.FC<{ url: string; children: React.ReactNode }> = ({
  url,
  children,
}) => (
  <button
    type="button"
    onClick={() => openUrl(url)}
    className="inline-flex items-center gap-1 text-xs font-medium text-logo-primary hover:underline"
  >
    {children}
    <ExternalLink className="w-3 h-3" />
  </button>
);

const Step: React.FC<{
  index: number;
  title: string;
  children: React.ReactNode;
}> = ({ index, title, children }) => (
  <li className="flex gap-3">
    <span className="flex items-center justify-center shrink-0 w-6 h-6 rounded-full bg-mid-gray/20 text-xs font-semibold">
      {index}
    </span>
    <div className="flex-1 min-w-0 space-y-1.5 pb-1">
      <p className="text-sm font-medium">{title}</p>
      {children}
    </div>
  </li>
);

/** Steps 1–2 of the walkthrough: create an account, then an API key. */
export const DeepgramKeySteps: React.FC = () => {
  const { t } = useTranslation();
  return (
    <>
      <Step index={1} title={t("cloudSetup.steps.account.title")}>
        <p className="text-xs text-text/60">
          {t("cloudSetup.steps.account.body")}
        </p>
        <LinkButton url={SIGNUP_URL}>
          {t("cloudSetup.steps.account.link")}
        </LinkButton>
      </Step>
      <Step index={2} title={t("cloudSetup.steps.key.title")}>
        <ol className="list-decimal ps-4 text-xs text-text/60 space-y-0.5">
          <li>{t("cloudSetup.steps.key.openConsole")}</li>
          <li>{t("cloudSetup.steps.key.openSettings")}</li>
          <li>{t("cloudSetup.steps.key.create")}</li>
          <li>{t("cloudSetup.steps.key.copy")}</li>
        </ol>
        <div className="flex flex-wrap gap-3">
          <LinkButton url={CONSOLE_URL}>
            {t("cloudSetup.steps.key.consoleLink")}
          </LinkButton>
          <LinkButton url={KEY_DOCS_URL}>
            {t("cloudSetup.steps.key.docsLink")}
          </LinkButton>
        </div>
      </Step>
    </>
  );
};

export const DeepgramModelPicker: React.FC<{
  value: string;
  onChange: (model: string) => void;
  disabled?: boolean;
}> = ({ value, onChange, disabled = false }) => {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-1.5">
      {DEEPGRAM_MODELS.map((model) => {
        const active = value === model.id;
        return (
          <button
            key={model.id}
            type="button"
            disabled={disabled}
            onClick={() => onChange(model.id)}
            className={`flex items-baseline justify-between gap-3 rounded-lg border px-3 py-2 text-start transition-colors disabled:opacity-50 ${
              active
                ? "border-logo-primary bg-logo-primary/10"
                : "border-mid-gray/30 hover:bg-mid-gray/10"
            }`}
          >
            <span className="text-sm font-medium font-mono">{model.id}</span>
            <span className="text-xs text-text/60">
              {t(`cloudSetup.models.${model.hintKey}`)}
            </span>
          </button>
        );
      })}
    </div>
  );
};

interface DeepgramSetupProps {
  /** Called after the key, model, and cloud toggle are saved. */
  onComplete: () => void;
  onBack: () => void;
  /** Debug previews show the walkthrough but never save. */
  readOnly?: boolean;
}

/**
 * Full first-run walkthrough: account, API key, key check, model choice. On
 * finish it saves the key and model and turns cloud transcription on.
 */
export const DeepgramSetup: React.FC<DeepgramSetupProps> = ({
  onComplete,
  onBack,
  readOnly = false,
}) => {
  const { t } = useTranslation();
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState(DEEPGRAM_DEFAULT_MODEL);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const keyCheck = useDeepgramKeyCheck();

  const finish = async () => {
    if (readOnly) return;
    setSaving(true);
    setSaveError(null);
    const steps = [
      () => commands.changeCloudSttApiKeySetting(DEEPGRAM, apiKey.trim()),
      () => commands.changeCloudSttModelSetting(DEEPGRAM, model),
      () => commands.changeCloudSttEnabledSetting(true),
    ];
    for (const step of steps) {
      const result = await step();
      if (result.status === "error") {
        setSaveError(result.error);
        setSaving(false);
        return;
      }
    }
    setSaving(false);
    onComplete();
  };

  return (
    <div className="space-y-4 text-start">
      <div className="space-y-1">
        <h2 className="text-base font-semibold">{t("cloudSetup.title")}</h2>
        <p className="text-sm text-text/60">{t("cloudSetup.intro")}</p>
      </div>

      <ol className="space-y-4">
        <DeepgramKeySteps />
        <Step index={3} title={t("cloudSetup.steps.paste.title")}>
          <div className="flex gap-2">
            <Input
              type="password"
              value={apiKey}
              onChange={(event) => {
                setApiKey(event.target.value);
                keyCheck.reset();
              }}
              placeholder={t("settings.cloudTranscription.apiKey.placeholder")}
              className="flex-1"
              autoComplete="off"
              spellCheck={false}
            />
            <Button
              variant="secondary"
              size="sm"
              disabled={!apiKey.trim() || keyCheck.state.status === "checking"}
              onClick={() => keyCheck.check(apiKey)}
            >
              {t("cloudSetup.key.check")}
            </Button>
          </div>
          <KeyCheckStatus state={keyCheck.state} />
        </Step>
        <Step index={4} title={t("cloudSetup.steps.model.title")}>
          <DeepgramModelPicker value={model} onChange={setModel} />
        </Step>
      </ol>

      <p className="text-xs text-text/50">{t("cloudSetup.privacy")}</p>
      {saveError && <p className="text-xs text-red-400">{saveError}</p>}

      <div className="flex items-center justify-between pt-1">
        <Button variant="ghost" size="sm" onClick={onBack} disabled={saving}>
          {t("cloudSetup.back")}
        </Button>
        <Button
          variant="primary"
          size="md"
          disabled={keyCheck.state.status !== "valid" || saving}
          onClick={finish}
        >
          {saving ? t("cloudSetup.saving") : t("cloudSetup.finish")}
        </Button>
      </div>
    </div>
  );
};
