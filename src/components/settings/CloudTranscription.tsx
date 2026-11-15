import React from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "../../hooks/useSettings";
import {
  DEEPGRAM,
  DEEPGRAM_DEFAULT_MODEL,
  DeepgramKeySteps,
  DeepgramModelPicker,
  KeyCheckStatus,
  useCloudKeyCheck,
} from "../cloud/DeepgramSetup";
import { ApiKeyField } from "./PostProcessingSettingsApi/ApiKeyField";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { SettingContainer } from "../ui/SettingContainer";
import { SettingsGroup } from "../ui/SettingsGroup";
import { ToggleSwitch } from "../ui/ToggleSwitch";

/** Any OpenAI-compatible `/audio/transcriptions` server (self-hosted models). */
const CUSTOM = "custom";

const PROVIDERS: ReadonlyArray<{ id: string; hintKey: string }> = [
  { id: DEEPGRAM, hintKey: "deepgram" },
  { id: CUSTOM, hintKey: "custom" },
];

/** Text input that saves on blur, like the API key field. */
const BlurTextField: React.FC<{
  value: string;
  onCommit: (value: string) => void;
  disabled: boolean;
  placeholder: string;
}> = ({ value, onCommit, disabled, placeholder }) => {
  const [local, setLocal] = React.useState(value);
  React.useEffect(() => setLocal(value), [value]);
  return (
    <Input
      type="text"
      value={local}
      onChange={(event) => setLocal(event.target.value)}
      onBlur={() => {
        if (local.trim() !== value) onCommit(local.trim());
      }}
      placeholder={placeholder}
      variant="compact"
      disabled={disabled}
      spellCheck={false}
      className="flex-1 min-w-[320px] font-mono"
    />
  );
};

/**
 * Cloud speech-to-text. When enabled, audio goes to a hosted STT API instead
 * of the local engine: Deepgram (streams, custom words become keyterms), or a
 * custom OpenAI-compatible endpoint such as a self-hosted model server
 * (transcribes once at key release, optionally previewing while recording).
 */
export const CloudTranscription: React.FC = React.memo(() => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const enabled = getSetting("cloud_stt_enabled") || false;
  const provider = getSetting("cloud_stt_provider_id") || DEEPGRAM;
  const apiKeys = getSetting("cloud_stt_api_keys") || {};
  const models = getSetting("cloud_stt_models") || {};
  const baseUrls = getSetting("cloud_stt_base_urls") || {};
  const customPreview = getSetting("cloud_stt_custom_preview") ?? true;
  const isCustom = provider === CUSTOM;

  const keyCheck = useCloudKeyCheck(provider);
  const [showGuide, setShowGuide] = React.useState(false);

  const apiKey = apiKeys[provider] ?? "";
  const model = isCustom
    ? (models[CUSTOM] ?? "")
    : models[DEEPGRAM] || DEEPGRAM_DEFAULT_MODEL;
  const baseUrl = baseUrls[CUSTOM] ?? "";

  const setProvider = (value: string) => {
    if (value === provider) return;
    keyCheck.reset();
    updateSetting("cloud_stt_provider_id", value);
  };

  const setApiKey = (value: string) => {
    if (value.trim() === apiKey) return;
    keyCheck.reset();
    updateSetting("cloud_stt_api_keys", {
      ...apiKeys,
      [provider]: value.trim(),
    });
  };

  const setModel = (value: string) => {
    keyCheck.reset();
    updateSetting("cloud_stt_models", { ...models, [provider]: value });
  };

  const setBaseUrl = (value: string) => {
    keyCheck.reset();
    updateSetting("cloud_stt_base_urls", { ...baseUrls, [CUSTOM]: value });
  };

  // Deepgram needs a key to check; a custom endpoint needs a URL (its key is
  // optional).
  const canCheck = isCustom ? !!baseUrl : !!apiKey;

  return (
    <SettingsGroup title={t("settings.cloudTranscription.title")}>
      <ToggleSwitch
        checked={enabled}
        onChange={(next) => updateSetting("cloud_stt_enabled", next)}
        isUpdating={isUpdating("cloud_stt_enabled")}
        label={t("settings.cloudTranscription.toggle.label")}
        description={t("settings.cloudTranscription.toggle.description")}
        descriptionMode="tooltip"
        grouped={true}
      />

      {enabled && (
        <>
          <SettingContainer
            title={t("settings.cloudTranscription.provider.title")}
            description={t("settings.cloudTranscription.provider.description")}
            descriptionMode="tooltip"
            layout="stacked"
            grouped={true}
          >
            <div className="flex flex-col gap-1.5">
              {PROVIDERS.map((option) => {
                const active = provider === option.id;
                return (
                  <button
                    key={option.id}
                    type="button"
                    disabled={isUpdating("cloud_stt_provider_id")}
                    onClick={() => setProvider(option.id)}
                    className={`flex items-baseline justify-between gap-3 rounded-lg border px-3 py-2 text-start transition-colors disabled:opacity-50 ${
                      active
                        ? "border-logo-primary bg-logo-primary/10"
                        : "border-mid-gray/30 hover:bg-mid-gray/10"
                    }`}
                  >
                    <span className="text-sm font-medium">
                      {t(
                        `settings.cloudTranscription.provider.${option.hintKey}.label`,
                      )}
                    </span>
                    <span className="text-xs text-text/60">
                      {t(
                        `settings.cloudTranscription.provider.${option.hintKey}.hint`,
                      )}
                    </span>
                  </button>
                );
              })}
            </div>
          </SettingContainer>

          {isCustom && (
            <SettingContainer
              title={t("settings.cloudTranscription.customUrl.title")}
              description={t(
                "settings.cloudTranscription.customUrl.description",
              )}
              descriptionMode="tooltip"
              layout="stacked"
              grouped={true}
            >
              <BlurTextField
                value={baseUrl}
                onCommit={setBaseUrl}
                disabled={isUpdating("cloud_stt_base_urls")}
                placeholder={t(
                  "settings.cloudTranscription.customUrl.placeholder",
                )}
              />
            </SettingContainer>
          )}

          <SettingContainer
            title={t(
              isCustom
                ? "settings.cloudTranscription.customKey.title"
                : "settings.cloudTranscription.apiKey.title",
            )}
            description={t(
              isCustom
                ? "settings.cloudTranscription.customKey.description"
                : "settings.cloudTranscription.apiKey.description",
            )}
            descriptionMode="tooltip"
            layout="stacked"
            grouped={true}
          >
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <ApiKeyField
                  value={apiKey}
                  onBlur={setApiKey}
                  disabled={isUpdating("cloud_stt_api_keys")}
                  placeholder={t(
                    isCustom
                      ? "settings.cloudTranscription.customKey.placeholder"
                      : "settings.cloudTranscription.apiKey.placeholder",
                  )}
                />
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={!canCheck || keyCheck.state.status === "checking"}
                  onClick={() => keyCheck.check(apiKey)}
                >
                  {t(
                    isCustom
                      ? "settings.cloudTranscription.customKey.test"
                      : "cloudSetup.key.check",
                  )}
                </Button>
              </div>
              <KeyCheckStatus state={keyCheck.state} />
              {!isCustom && (
                <>
                  <button
                    type="button"
                    onClick={() => setShowGuide((v) => !v)}
                    className="self-start text-xs font-medium text-text/60 hover:text-text"
                  >
                    {showGuide
                      ? t("cloudSetup.hideGuide")
                      : t("cloudSetup.showGuide")}
                  </button>
                  {showGuide && (
                    <ol className="space-y-3 pt-1">
                      <DeepgramKeySteps />
                    </ol>
                  )}
                </>
              )}
            </div>
          </SettingContainer>

          <SettingContainer
            title={t("settings.cloudTranscription.model.title")}
            description={t(
              isCustom
                ? "settings.cloudTranscription.customModel.description"
                : "settings.cloudTranscription.model.description",
            )}
            descriptionMode="tooltip"
            layout="stacked"
            grouped={true}
          >
            {isCustom ? (
              <BlurTextField
                value={model}
                onCommit={setModel}
                disabled={isUpdating("cloud_stt_models")}
                placeholder={t(
                  "settings.cloudTranscription.customModel.placeholder",
                )}
              />
            ) : (
              <DeepgramModelPicker
                value={model}
                onChange={setModel}
                disabled={isUpdating("cloud_stt_models")}
              />
            )}
          </SettingContainer>

          {isCustom && (
            <ToggleSwitch
              checked={customPreview}
              onChange={(next) =>
                updateSetting("cloud_stt_custom_preview", next)
              }
              isUpdating={isUpdating("cloud_stt_custom_preview")}
              label={t("settings.cloudTranscription.customPreview.label")}
              description={t(
                "settings.cloudTranscription.customPreview.description",
              )}
              descriptionMode="tooltip"
              grouped={true}
            />
          )}
        </>
      )}
    </SettingsGroup>
  );
});

CloudTranscription.displayName = "CloudTranscription";
