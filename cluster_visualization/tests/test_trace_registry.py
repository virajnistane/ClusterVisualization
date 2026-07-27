"""Tests for the centralized trace registry module."""

import pytest
from cluster_visualization.src.visualization.trace_registry import (
    TraceRegistry,
    TraceType,
    TRACE_SPECS,
    _get_trace_name,
)


# ---------------------------------------------------------------------------
# Sample trace dicts
# ---------------------------------------------------------------------------

SAMPLE_TRACES = {
    "catred_masked": {"name": "CATRED Masked - MER Tile", "type": "scattergl", "x": [1], "y": [2]},
    "catred_unmasked": {"name": "CATRED Unmasked - Boxed", "type": "scattergl", "x": [3], "y": [4]},
    "mosaic_image": {"name": "Mosaic (CDS) 12345", "type": "image", "source": "data:image/png;base64,..."},
    "mosaic_heatmap": {"name": "MER-Mosaic cutout #1", "type": "heatmap", "z": [[1, 2], [3, 4]]},
    "mosaic_placeholder": {"name": "Mosaic 12345 (Placeholder)", "type": "heatmap", "z": [[0]]},
    "mask_overlay_bin": {"name": "Mask overlay bin 3", "type": "scatter", "x": [1], "y": [2]},
    "inverted_mask": {"name": "Inverted mask overlay", "type": "scatter", "x": [], "y": []},
    "mask_colorbar": {"name": "Mask Colorbar", "type": "scatter", "marker": {"colorscale": "viridis"}},
    "mask_aladin": {"name": "Mask aladin moc", "type": "scatter", "customdata": [1, 2, 3]},
    "inverted_mask_aladin": {"name": "Inverted mask aladin moc", "type": "scatter"},
    "polygon_lev1": {"name": "Tile 100 LEV1", "type": "scatter", "x": [1], "y": [2]},
    "polygon_core": {"name": "Tile 100 CORE", "type": "scatter", "x": [3], "y": [4]},
    "polygon_mertile": {"name": "MER-Tile 200", "type": "scatter", "x": [5], "y": [6]},
    "members": {"name": "Members (ID 42)", "type": "scattergl", "x": [7], "y": [8]},
    "matched_pair": {"name": "Matched Pair", "type": "scatter", "x": [9], "y": [10]},
    "selected": {"name": "__selected_cluster__", "type": "scattergl", "x": [11], "y": [12]},
    "merged_pzwav": {"name": "Merged PZWAV", "type": "scattergl", "x": [13], "y": [14]},
    "merged_amico": {"name": "Merged AMICO", "type": "scattergl", "x": [15], "y": [16]},
    "unmerged_tile": {"name": "PZWAV Unmerged-Tile 100", "type": "scattergl"},
    "enhanced": {"name": "PZWAV (Enhanced)", "type": "scattergl"},
    "proximity": {"name": "Cluster in Proximity", "type": "scattergl"},
    "unnamed": {"type": "scatter", "x": [0], "y": [0]},
}


# ---------------------------------------------------------------------------
# Test match predicates
# ---------------------------------------------------------------------------

class TestClassification:
    def test_catred_traces(self):
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["catred_masked"]) == TraceType.CATRED
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["catred_unmasked"]) == TraceType.CATRED

    def test_mosaic_traces(self):
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["mosaic_image"]) == TraceType.MOSAIC
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["mosaic_heatmap"]) == TraceType.MOSAIC
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["mosaic_placeholder"]) == TraceType.MOSAIC

    def test_mask_overlay_traces(self):
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["mask_overlay_bin"]) == TraceType.MASK_OVERLAY
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["inverted_mask"]) == TraceType.MASK_OVERLAY
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["mask_colorbar"]) == TraceType.MASK_OVERLAY
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["mask_aladin"]) == TraceType.MASK_OVERLAY
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["inverted_mask_aladin"]) == TraceType.MASK_OVERLAY

    def test_polygon_traces(self):
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["polygon_lev1"]) == TraceType.POLYGON
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["polygon_core"]) == TraceType.POLYGON
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["polygon_mertile"]) == TraceType.POLYGON

    def test_members_traces(self):
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["members"]) == TraceType.MEMBERS

    def test_matched_pair(self):
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["matched_pair"]) == TraceType.MATCHED_PAIR

    def test_selected_cluster(self):
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["selected"]) == TraceType.SELECTED_CLUSTER

    def test_cluster_traces(self):
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["merged_pzwav"]) == TraceType.CLUSTER
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["merged_amico"]) == TraceType.CLUSTER
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["unmerged_tile"]) == TraceType.CLUSTER
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["enhanced"]) == TraceType.CLUSTER
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["proximity"]) == TraceType.CLUSTER

    def test_unnamed_trace_returns_none(self):
        assert TraceRegistry.classify_trace(SAMPLE_TRACES["unnamed"]) is None

    def test_empty_name_returns_none(self):
        assert TraceRegistry.classify_trace({"name": "", "type": "scatter"}) is None


# ---------------------------------------------------------------------------
# Test extract_traces
# ---------------------------------------------------------------------------

class TestExtractTraces:
    def _make_figure(self, trace_keys):
        return {"data": [SAMPLE_TRACES[k] for k in trace_keys]}

    def test_extract_catred_only(self):
        fig = self._make_figure(["catred_masked", "mosaic_image", "merged_pzwav"])
        result = TraceRegistry.extract_traces(fig, {TraceType.CATRED})
        assert len(result[TraceType.CATRED]) == 1

    def test_extract_multiple_types(self):
        fig = self._make_figure([
            "catred_masked", "mosaic_image", "mask_colorbar", "merged_pzwav"
        ])
        result = TraceRegistry.extract_traces(
            fig, {TraceType.CATRED, TraceType.MOSAIC, TraceType.MASK_OVERLAY}
        )
        assert len(result[TraceType.CATRED]) == 1
        assert len(result[TraceType.MOSAIC]) == 1
        assert len(result[TraceType.MASK_OVERLAY]) == 1

    def test_extract_empty_figure(self):
        result = TraceRegistry.extract_traces(None, {TraceType.CATRED})
        assert result[TraceType.CATRED] == []

    def test_extract_no_matching(self):
        fig = self._make_figure(["merged_pzwav"])
        result = TraceRegistry.extract_traces(fig, {TraceType.CATRED})
        assert result[TraceType.CATRED] == []


# ---------------------------------------------------------------------------
# Test extract_all_preserved
# ---------------------------------------------------------------------------

class TestExtractAllPreserved:
    def test_excludes_specified_types(self):
        fig = {"data": [
            SAMPLE_TRACES["catred_masked"],
            SAMPLE_TRACES["mosaic_image"],
            SAMPLE_TRACES["merged_pzwav"],
            SAMPLE_TRACES["polygon_lev1"],
        ]}
        result = TraceRegistry.extract_all_preserved(
            fig, exclude={TraceType.CATRED, TraceType.CLUSTER}
        )
        assert TraceType.CATRED not in result or len(result.get(TraceType.CATRED, [])) == 0
        assert TraceType.CLUSTER not in result or len(result.get(TraceType.CLUSTER, [])) == 0
        assert len(result[TraceType.MOSAIC]) == 1
        assert len(result[TraceType.POLYGON]) == 1


# ---------------------------------------------------------------------------
# Test assemble_in_layer_order
# ---------------------------------------------------------------------------

class TestAssembleInLayerOrder:
    def test_correct_ordering(self):
        categorized = {
            TraceType.CLUSTER: [SAMPLE_TRACES["merged_pzwav"]],
            TraceType.CATRED: [SAMPLE_TRACES["catred_masked"]],
            TraceType.POLYGON: [SAMPLE_TRACES["polygon_lev1"]],
            TraceType.MOSAIC: [SAMPLE_TRACES["mosaic_image"]],
            TraceType.MASK_OVERLAY: [SAMPLE_TRACES["mask_colorbar"]],
        }
        ordered = TraceRegistry.assemble_in_layer_order(categorized)

        names = [t.get("name", "") if isinstance(t, dict) else t.name for t in ordered]
        polygon_idx = names.index("Tile 100 LEV1")
        mosaic_idx = names.index("Mosaic (CDS) 12345")
        mask_idx = names.index("Mask Colorbar")
        catred_idx = names.index("CATRED Masked - MER Tile")
        cluster_idx = names.index("Merged PZWAV")

        assert polygon_idx < mosaic_idx < mask_idx < catred_idx < cluster_idx

    def test_empty_categories(self):
        categorized = {TraceType.POLYGON: [], TraceType.CLUSTER: [SAMPLE_TRACES["merged_pzwav"]]}
        ordered = TraceRegistry.assemble_in_layer_order(categorized)
        assert len(ordered) == 1


# ---------------------------------------------------------------------------
# Regression: mask bug
# ---------------------------------------------------------------------------

class TestMaskOverlayBugRegression:
    """Ensure 'Mask Colorbar' and 'Inverted mask overlay' are classified as MASK_OVERLAY."""

    def test_mask_colorbar_classified(self):
        assert TraceRegistry.classify_trace({"name": "Mask Colorbar"}) == TraceType.MASK_OVERLAY

    def test_inverted_mask_classified(self):
        assert TraceRegistry.classify_trace({"name": "Inverted mask overlay"}) == TraceType.MASK_OVERLAY

    def test_inverted_mask_aladin_classified(self):
        assert TraceRegistry.classify_trace({"name": "Inverted mask aladin moc"}) == TraceType.MASK_OVERLAY
