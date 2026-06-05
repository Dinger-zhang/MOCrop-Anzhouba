from mocrop import data_loader


def test_missing_demo_data_is_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "DATA_DIR", tmp_path)

    bundle = data_loader.load_data_bundle()

    assert len(bundle.parcels) == 6
    assert len(bundle.crops) == 12
    assert (tmp_path / "parcels.csv").exists()
    assert (tmp_path / "crops.csv").exists()
    assert (tmp_path / "homestays.csv").exists()
