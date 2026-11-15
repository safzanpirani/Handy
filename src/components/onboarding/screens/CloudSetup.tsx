import React from "react";
import { DeepgramSetup } from "../../cloud/DeepgramSetup";
import { OnboardingShell } from "../OnboardingShell";

interface CloudSetupProps {
  progress?: { index: number; count: number };
  /** Debug preview: the walkthrough shows but never saves. */
  preview?: boolean;
  onBack: () => void;
  /** Called once the key and model are saved and cloud transcription is on. */
  onComplete: () => void;
}

/** Model's alternative: transcribe through Deepgram, with nothing to download. */
export const CloudSetup: React.FC<CloudSetupProps> = ({
  progress,
  preview = false,
  onBack,
  onComplete,
}) => (
  <OnboardingShell progress={progress}>
    <div className="h-full overflow-y-auto px-gutter py-2 select-text cursor-auto">
      <DeepgramSetup
        readOnly={preview}
        onBack={onBack}
        onComplete={onComplete}
      />
    </div>
  </OnboardingShell>
);
