import React from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "../../hooks/useSettings";
import { ApiKeyField } from "./PostProcessingSettingsApi/ApiKeyField";
import { Input } from "../ui/Input";
import { SettingContainer } from "../ui/SettingContainer";
import { SettingsGroup } from "../ui/SettingsGroup";
import { ToggleSwitch } from "../ui/ToggleSwitch";

const DEEPGRAM = "deepgram";

/**
 * Cloud speech-to-text. When enabled, audio is sent to a hosted STT API
 * instead of the local engine — no model download, but every dictation needs
 * network. Custom words are forwarded as decode-time bias where supported.
 */
export const CloudTranscription: React.FC = React.memo(() => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const enabled = getSetting("cloud_stt_enabled") || false;
  const apiKeys = getSetting("cloud_stt_api_keys") || {};
  const models = getSetting("cloud_stt_models") || {};

  const setApiKey = (value: string) => {
    updateSetting("cloud_stt_api_keys", { ...apiKeys, [DEEPGRAM]: value });
  };

  const storedModel = models[DEEPGRAM] ?? "";
  const [modelDraft, setModelDraft] = React.useState(storedModel);
  React.useEffect(() => setModelDraft(storedModel), [storedModel]);

  const setModel = (value: string) => {
    updateSetting("cloud_stt_models", { ...models, [DEEPGRAM]: value.trim() });
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
            layout="horizontal"
            grouped={true}
          >
            <ApiKeyField
              value={apiKeys[DEEPGRAM] ?? ""}
              onBlur={setApiKey}
              disabled={isUpdating("cloud_stt_api_keys")}
              placeholder={t("settings.cloudTranscription.apiKey.placeholder")}
            />
          </SettingContainer>

          <SettingContainer
            title={t("settings.cloudTranscription.model.title")}
            description={t("settings.cloudTranscription.model.description")}
            descriptionMode="tooltip"
            layout="horizontal"
            grouped={true}
          >
            <Input
              type="text"
              value={modelDraft}
              onChange={(event) => setModelDraft(event.target.value)}
              onBlur={() => setModel(modelDraft)}
              placeholder="nova-3"
              variant="compact"
              disabled={isUpdating("cloud_stt_models")}
              className="min-w-[200px]"
            />
          </SettingContainer>
        </>
      )}
    </SettingsGroup>
  );
});

CloudTranscription.displayName = "CloudTranscription";
