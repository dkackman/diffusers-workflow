"""Stars: the "keep" mark on an output file, in the run's notes.json beside
its notes (dw/notes.py), set through PUT /api/stars and carried on the
gallery listing."""

from dw import notes as N

from .test_notes_api import (  # noqa: F401 - fixtures
    client,
    make_run,
    output_root,
    run_name,
    studio,
)


def test_star_round_trip_keeps_notes(tmp_path):
    N.set_file_note(str(tmp_path), "a.png", "keeper")
    assert N.set_file_starred(str(tmp_path), "a.png", True) is True
    assert N.read_starred(str(tmp_path)) == {"a.png"}
    # A note written after the star leaves it, and the other way round
    N.set_run_note(str(tmp_path), "run note")
    assert N.read_starred(str(tmp_path)) == {"a.png"}
    assert N.read_notes(str(tmp_path))["files"] == {"a.png": "keeper"}


def test_unstarring_the_last_file_with_no_notes_removes_the_sidecar(tmp_path):
    N.set_file_starred(str(tmp_path), "a.png", True)
    N.set_file_starred(str(tmp_path), "a.png", False)
    assert not (tmp_path / N.NOTES_FILE_NAME).exists()


def test_dropping_a_file_drops_its_star(tmp_path):
    N.set_file_starred(str(tmp_path), "a.png", True)
    N.drop_file_note(str(tmp_path), "a.png")
    assert N.read_starred(str(tmp_path)) == set()


def test_put_star_and_the_listing_carries_it(client, output_root, make_run):
    run_dir = make_run("wf", ["a.png", "b.png"])
    run = run_name(run_dir, output_root)
    r = client.put("/api/stars", json={"name": f"{run}/a.png", "starred": True})
    assert r.status_code == 200
    assert r.json() == {"name": f"{run}/a.png", "starred": True}
    files = {f["label"]: f for f in client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["starred"] is True
    assert "starred" not in files["b.png"]
    client.put("/api/stars", json={"name": f"{run}/a.png", "starred": False})
    files = {f["label"]: f for f in client.get("/api/gallery").json()["files"]}
    assert "starred" not in files["a.png"]


def test_only_an_output_file_can_be_starred(client, output_root, make_run):
    run = run_name(make_run("wf", ["a.png"]), output_root)
    for name in (run, "workspace:default", "asset:x.png"):
        r = client.put("/api/stars", json={"name": name, "starred": True})
        assert r.status_code == 400, name


def test_put_star_refuses_traversal(client):
    r = client.put("/api/stars", json={"name": "../../etc/passwd", "starred": True})
    assert r.status_code in (400, 404)
