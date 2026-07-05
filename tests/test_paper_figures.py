from paper_figures import generate_paper_figures


def test_paper_figures_generate_svg_artifacts():
    outputs = generate_paper_figures()
    expected = {
        "dte_mechanism",
        "layered_memory",
        "feasibility_allocation",
        "semiconductor_topology",
        "choice_relocation",
        "topology_surgery",
        "feedback_continuum",
        "model_benchmark",
        "topology_nulls",
    }
    assert expected == set(outputs)
    for path in outputs.values():
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert "<svg" in text
        assert "</svg>" in text
