import React from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "../../hooks/useSettings";
import {
  DEEPGRAM,
  DEEPGRAM_DEFAULT_MODEL,
  DeepgramKeySteps,
  DeepgramModelPicker,
  KeyCheckStatus,
  useDeepgramKeyCheck,
} from "../cloud/DeepgramSetup";
import { ApiKeyField } from "./PostProcessingSettingsApi/ApiKeyField";
import { Button } from "../ui/Button";
import { SettingContainer } from "../ui/SettingContainer";
import { SettingsGroup } from "../ui/SettingsGroup";
import { ToggleSwitch } from "../ui/ToggleSwitch";

/**
 * Cloud speech-to-text. When enabled, audio streams to Deepgram instead of the
 * local engine: no model download, but every dictation needs network. Custom
 * words are forwarded as decode-time keyterms.
 */
export const CloudTranscription: React.FC = React.memo(() => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const keyCheck = useDeepgramKeyCheck();
  const [showGuide, setShowGuide] = React.useState(false);

  const enabled = getSetting("cloud_stt_enabled") || false;
  const apiKeys = getSetting("cloud_stt_api_keys") || {};
  const models = getSetting("cloud_stt_models") || {};
  const apiKey = apiKeys[DEEPGRAM] ?? "";
  const model = models[DEEPGRAM] || DEEPGRAM_DEFAULT_MODEL;

  const setApiKey = (value: string) => {
    if (value.trim() === apiKey) return;
    keyCheck.reset();
    updateSetting("cloud_stt_api_keys", {
      ...apiKeys,
      [DEEPGRAM]: value.trim(),
    });
  };

  const setModel = (value: string) => {
    updateSetting("cloud_stt_models", { ...models, [DEEPGRAM]: value });
  };

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
            title={t("settings.cloudTranscription.apiKey.title")}
            description={t("settings.cloudTranscription.apiKey.description")}
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
                    "settings.cloudTranscription.apiKey.placeholder",
                  )}
                />
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={!apiKey || keyCheck.state.status === "checking"}
                  onClick={() => keyCheck.check(apiKey)}
                >
                  {t("cloudSetup.key.check")}
                </Button>
              </div>
              <KeyCheckStatus state={keyCheck.state} />
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
            </div>
          </SettingContainer>

          <SettingContainer
            title={t("settings.cloudTranscription.model.title")}
            description={t("settings.cloudTranscription.model.description")}
            descriptionMode="tooltip"
            layout="stacked"
            grouped={true}
          >
            <DeepgramModelPicker
              value={model}
              onChange={setModel}
              disabled={isUpdating("cloud_stt_models")}
            />
          </SettingContainer>
        </>
      )}
    </SettingsGroup>
  );
});

CloudTranscription.displayName = "CloudTranscription";
