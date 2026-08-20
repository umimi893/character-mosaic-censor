from __future__ import annotations

from pathlib import Path

from ..types import ProcessResult
from .main_window import MainWindow


_DEGRADED_ANALYSIS_PREFIXES = (
    "partial:",
    "partial_disabled:",
    "unavailable:",
    "failed:",
    "disabled_after_failure",
)


class EnhancedMainWindow(MainWindow):
    """Main window UX additions layered over the stable worker implementation."""

    def __init__(self):
        super().__init__(); open_input=getattr(self.controls,"open_input_requested",None)
        if open_input is not None: open_input.connect(self.open_input)
        self._relabel_manual_review_button()

    def _on_language_changed(self, language: str) -> None:
        super()._on_language_changed(language); self._relabel_manual_review_button()

    def _relabel_manual_review_button(self) -> None:
        button=getattr(self.controls,"open_manual_review_button",None)
        if button is None: return
        button.setText(self._t("手動修正用画像を開く","Open images for manual correction"))
        button.setToolTip(self._t("BBoxもモザイクも入っていない元画像を開きます。reference_bboxは検出位置確認用、auto_censoredは自動処理結果の比較用です。","Opens untouched originals with no BBox or censoring. reference_bbox is for detector reference and auto_censored is for comparison."))

    def open_input(self) -> None:
        text=self.controls.input_edit.text().strip()
        if text: self._open_local_path(Path(text))
    def open_manual_review(self) -> None:
        self.controls.ensure_output_default(); text=self.controls.output_edit.text().strip()
        if text: self._open_local_path(Path(text)/"_manual_review"/"edit")

    def _on_progress(self,index:int,total:int,src:str,result:ProcessResult)->None:
        self.progress.setRange(0,max(1,total)); self.progress.setValue(index); self.counter_label.setText(f"{index} / {total}")
        if result.error: self.detect_label.setText(f"ERROR: {Path(src).name}"); self.statusBar().showMessage(result.error); return
        if result.cancelled:
            self.detect_label.setText(f"CANCEL: {Path(src).name}"); self.statusBar().showMessage(self._t("停止しました — 処理途中の画像は保存していません","Stopped — the incomplete image was not saved")); return
        if result.skipped: self.detect_label.setText(f"SKIP: {Path(src).name}"); return
        suppressed=len(result.anatomy_suppressed); evidence_review=sum(1 for item in result.candidate_evidence if item.decision=="review")
        filtered=self._t(f" / 人体解析で除外 {suppressed}",f" / body-filtered {suppressed}") if suppressed else ""
        body_review=self._t(f" / 人体解析Review {evidence_review}",f" / body-review {evidence_review}") if evidence_review else ""
        status=result.anatomy_filter_status; degraded=status.startswith(_DEGRADED_ANALYSIS_PREFIXES); degraded_text=self._t(" / 人体解析一部無効"," / body analysis degraded") if degraded else ""
        if result.detections:
            top=max(result.detections,key=lambda d:d.score); review=" / Review" if result.review_required else ""; over=self._t(" / 検出過多"," / Over-detected") if result.count_mismatch else ""
            self.detect_label.setText(f"Detected: {len(result.detections)} / {top.label} {top.score:.2f}{review}{over}{filtered}{body_review}{degraded_text}")
        else:
            review=" / Review" if result.review_required else ""; self.detect_label.setText(self._t(f"対象未検出{review}{filtered}{body_review}{degraded_text}",f"No target detected{review}{filtered}{body_review}{degraded_text}"))
        if degraded:
            self.statusBar().showMessage(self._t(f"人体解析の一部を利用できません ({status})。安全側で通常検出を維持しています。",f"Part of body analysis is unavailable ({status}); normal detection is kept fail-open."))

    def _on_finished(self,results:list[ProcessResult],log_path:str,stopped:bool)->None:
        self._last_log_path=Path(log_path); errors=sum(1 for r in results if r.error); reviews=sum(1 for r in results if r.review_required); over=sum(1 for r in results if r.count_mismatch)
        detected=sum(1 for r in results if r.detections and not r.error and not r.cancelled and not r.skipped); no_target=sum(1 for r in results if not r.detections and not r.error and not r.cancelled and not r.skipped); skipped=sum(1 for r in results if r.skipped)
        suppressed=sum(len(r.anatomy_suppressed) for r in results); body_review=sum(sum(1 for item in r.candidate_evidence if item.decision=="review") for r in results); degraded=sum(1 for r in results if r.anatomy_filter_status.startswith(_DEGRADED_ANALYSIS_PREFIXES))
        state=self._t("停止","Stopped") if stopped else self._t("完了","Complete")
        summary=self._t(f"{state}: {len(results)}件 / 検出あり {detected}件 / 対象未検出 {no_target}件 / 人体解析で除外 {suppressed}件 / 人体解析Review {body_review}件 / 人体解析縮退 {degraded}件 / 検出過多 {over}件 / Review {reviews}件 / Skip {skipped}件 / エラー {errors}件\nログ: {log_path}",f"{state}: {len(results)} image(s) / target detected {detected} / no target detected {no_target} / body-filtered {suppressed} / body-review {body_review} / body-analysis degraded {degraded} / over-detected {over} / Review {reviews} / skipped {skipped} / errors {errors}\nLog: {log_path}")
        self.controls.set_summary(summary); self.statusBar().showMessage(summary.replace("\n","  ")); self.controls.set_running(False)
