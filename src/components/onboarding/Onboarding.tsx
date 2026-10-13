import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { ChevronDown, Cloud, SkipForward } from "lucide-react";
import { commands, type ModelInfo } from "@/bindings";
import type { ModelCardStatus } from "./ModelCard";
import ModelCard, { isLegacySource } from "./ModelCard";
import HandyTextLogo from "../icons/HandyTextLogo";
import { useModelStore } from "../../stores/modelStore";
import { useSettingsStore } from "../../stores/settingsStore";
import { DeepgramSetup } from "../cloud/DeepgramSetup";

interface OnboardingProps {
  onModelSelected: () => void;
  preview?: boolean;
}

const Onboarding: React.FC<OnboardingProps> = ({
  onModelSelected,
  preview = false,
}) => {
  const { t } = useTranslation();
  const {
    models,
    downloadModel,
    selectModel,
    downloadingModels,
    verifyingModels,
    extractingModels,
    downloadProgress,
    downloadStats,
    cancelDownload,
  } = useModelStore();
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const hasStartedSelection = useRef(false);
  const [mode, setMode] = useState<"models" | "cloud">("models");
  const [finishing, setFinishing] = useState(false);

  const isBusy = selectedModelId !== null;

  // Curate the download list: legacy (.bin/ONNX) downloads are deprecated and
  // never shown here (they still appear in the compatible section if already on
  // disk). The catalog arrives rank-sorted, so the first two recommended models
  // are the featured picks — currently Parakeet Unified (English) and Nemotron
  // Streaming (multilingual). Everything else hides behind "Show all".
  const { downloadable, topPicks, otherRecommended, rest } = useMemo(() => {
    const downloadable = models.filter(
      (m: ModelInfo) => !m.is_downloaded && !isLegacySource(m),
    );
    const recommended = downloadable.filter((m: ModelInfo) => m.is_recommended);
    // `models` arrives in editorial rank order (the backend sorts by rank_of,
    // then accuracy), so keep that order here: ranked-but-not-recommended models
    // surface first, then the unranked tail by accuracy.
    const rest = downloadable.filter((m: ModelInfo) => !m.is_recommended);
    return {
      downloadable,
      topPicks: recommended.slice(0, 2),
      otherRecommended: recommended.slice(2),
      rest,
    };
  }, [models]);

  const hasRecommended = topPicks.length > 0 || otherRecommended.length > 0;
  // When nothing recommended remains to download (e.g. all already on disk),
  // there is no curated subset to collapse, so just show the full list.
  const showRest = showAll || !hasRecommended;

  // Watch for the selected model to finish downloading + verifying + extracting
  useEffect(() => {
    // Debug previews are inert: never switch the user's active model. Guarded
    // here as well as in the handlers because this is where the backend call
    // actually happens.
    if (preview) return;

    if (!selectedModelId) {
      hasStartedSelection.current = false;
      return;
    }

    const model = models.find((m) => m.id === selectedModelId);
    const stillDownloading = selectedModelId in downloadingModels;
    const stillVerifying = selectedModelId in verifyingModels;
    const stillExtracting = selectedModelId in extractingModels;

    if (
      model?.is_downloaded &&
      !stillDownloading &&
      !stillVerifying &&
      !stillExtracting &&
      !hasStartedSelection.current
    ) {
      hasStartedSelection.current = true;

      // Model is ready — select it and transition
      selectModel(selectedModelId).then((success) => {
        if (success) {
          onModelSelected();
        } else {
          toast.error(t("onboarding.errors.selectModel"));
          hasStartedSelection.current = false;
          setSelectedModelId(null);
        }
      });
    }
  }, [
    selectedModelId,
    models,
    downloadingModels,
    verifyingModels,
    extractingModels,
    selectModel,
    onModelSelected,
    preview,
    t,
  ]);

  const handleDownloadModel = async (modelId: string) => {
    if (preview) return;

    setSelectedModelId(modelId);

    // Error toast is handled centrally by the model-download-failed event listener
    // in modelStore — no toast here to avoid duplicates.
    const success = await downloadModel(modelId);
    if (!success) {
      setSelectedModelId(null);
    }
  };

  const handleCancelDownload = async (modelId: string) => {
    if (preview) return;

    const success = await cancelDownload(modelId);
    if (success) {
      setSelectedModelId(null);
    }
  };

  const handleSelectExistingModel = (modelId: string) => {
    if (preview) return;

    setSelectedModelId(modelId);
  };

  const getModelStatus = (modelId: string): ModelCardStatus => {
    if (modelId in extractingModels) return "extracting";
    if (modelId in verifyingModels) return "verifying";
    if (modelId in downloadingModels) return "downloading";
    return "downloadable";
  };

  const getExistingModelStatus = (modelId: string): ModelCardStatus => {
    if (selectedModelId === modelId) return "switching";
    return "available";
  };

  const getModelDownloadProgress = (modelId: string): number | undefined => {
    return downloadProgress[modelId]?.percentage;
  };

  const getModelDownloadSpeed = (modelId: string): number | undefined => {
    return downloadStats[modelId]?.speed;
  };

  /** Leave onboarding without a local model (cloud setup or "skip"). */
  const finishWithoutModel = async () => {
    if (preview) return;
    setFinishing(true);
    const result = await commands.finishOnboardingWithoutModel();
    if (result.status === "error") {
      toast.error(result.error);
      setFinishing(false);
      return;
    }
    await useSettingsStore.getState().refreshSettings();
    onModelSelected();
  };

  if (mode === "cloud") {
    return (
      <div className="h-screen w-full flex flex-col p-6 gap-4 overflow-y-auto">
        <div className="flex flex-col items-center gap-2 shrink-0">
          <HandyTextLogo width={160} />
        </div>
        <div className="max-w-[560px] w-full mx-auto pb-6">
          <DeepgramSetup
            readOnly={preview}
            onBack={() => setMode("models")}
            onComplete={finishWithoutModel}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full flex flex-col p-6 gap-4">
      <div className="flex flex-col items-center gap-2 shrink-0">
        <HandyTextLogo width={200} />
        <p className="text-text/70 max-w-md font-medium mx-auto">
          {t("onboarding.subtitle")}
        </p>
      </div>

      <div className="max-w-[600px] w-full mx-auto text-center flex-1 flex flex-col min-h-0">
        <div className="space-y-6 pb-6">
          {models.some((m: ModelInfo) => m.is_downloaded) && (
            <div className="space-y-3">
              <div className="text-left">
                <h2 className="text-sm font-medium text-text/60">
                  {t("onboarding.existingModelsTitle")}
                </h2>
              </div>
              {models
                .filter((m: ModelInfo) => m.is_downloaded)
                .map((model: ModelInfo) => (
                  <ModelCard
                    key={model.id}
                    model={model}
                    status={getExistingModelStatus(model.id)}
                    disabled={isBusy}
                    onSelect={handleSelectExistingModel}
                    showRecommended={false}
                  />
                ))}
            </div>
          )}

          {downloadable.length > 0 && (
            <div className="space-y-3">
              <div className="text-left">
                <h2 className="text-sm font-medium text-text/60">
                  {t("onboarding.downloadModelsTitle")}
                </h2>
              </div>

              {topPicks.map((model: ModelInfo) => (
                <ModelCard
                  key={model.id}
                  model={model}
                  variant="featured"
                  status={getModelStatus(model.id)}
                  disabled={isBusy}
                  onSelect={handleDownloadModel}
                  onDownload={handleDownloadModel}
                  onCancel={handleCancelDownload}
                  downloadProgress={getModelDownloadProgress(model.id)}
                  downloadSpeed={getModelDownloadSpeed(model.id)}
                  showRecommended={false}
                />
              ))}

              {otherRecommended.map((model: ModelInfo) => (
                <ModelCard
                  key={model.id}
                  model={model}
                  status={getModelStatus(model.id)}
                  disabled={isBusy}
                  onSelect={handleDownloadModel}
                  onDownload={handleDownloadModel}
                  onCancel={handleCancelDownload}
                  downloadProgress={getModelDownloadProgress(model.id)}
                  downloadSpeed={getModelDownloadSpeed(model.id)}
                  showRecommended={false}
                />
              ))}

              {hasRecommended && rest.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowAll((v) => !v)}
                  className="flex items-center justify-center gap-1.5 mx-auto py-1 text-sm font-medium text-text/60 hover:text-text transition-colors"
                >
                  {showAll
                    ? t("onboarding.showFewerModels")
                    : t("onboarding.showAllModels", {
                        total: downloadable.length,
                      })}
                  <ChevronDown
                    className={`w-4 h-4 transition-transform duration-200 ${
                      showAll ? "rotate-180" : ""
                    }`}
                  />
                </button>
              )}

              {showRest &&
                rest.map((model: ModelInfo) => (
                  <ModelCard
                    key={model.id}
                    model={model}
                    status={getModelStatus(model.id)}
                    disabled={isBusy}
                    onSelect={handleDownloadModel}
                    onDownload={handleDownloadModel}
                    onCancel={handleCancelDownload}
                    downloadProgress={getModelDownloadProgress(model.id)}
                    downloadSpeed={getModelDownloadSpeed(model.id)}
                    showRecommended={false}
                  />
                ))}
            </div>
          )}

          <div className="space-y-3 pt-4 border-t border-mid-gray/20">
            <div className="text-left">
              <h2 className="text-sm font-medium text-text/60">
                {t("onboarding.noDownload.title")}
              </h2>
            </div>
            <button
              type="button"
              disabled={isBusy || finishing}
              onClick={() => setMode("cloud")}
              className="w-full flex items-start gap-3 rounded-xl border border-mid-gray/30 px-4 py-3 text-start hover:bg-mid-gray/10 transition-colors disabled:opacity-50"
            >
              <Cloud className="w-5 h-5 mt-0.5 shrink-0 text-logo-primary" />
              <span className="space-y-0.5">
                <span className="block text-sm font-semibold">
                  {t("onboarding.noDownload.cloud.title")}
                </span>
                <span className="block text-xs text-text/60">
                  {t("onboarding.noDownload.cloud.description")}
                </span>
              </span>
            </button>
            <button
              type="button"
              disabled={isBusy || finishing}
              onClick={finishWithoutModel}
              className="flex items-center gap-1.5 mx-auto py-1 text-sm font-medium text-text/60 hover:text-text transition-colors disabled:opacity-50"
            >
              <SkipForward className="w-4 h-4" />
              {t("onboarding.noDownload.skip")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Onboarding;
