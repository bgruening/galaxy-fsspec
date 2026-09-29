"""Unit tests for GalaxyFileSystem using a fake bioblend client (no network)."""

from __future__ import annotations

import pytest

from galaxy_fsspec.fs import GalaxyFileSystem


class FakeHistories:
    def __init__(self, store):
        self.store = store

    def get_histories(self):
        # Realistic: bioblend returns only id + name here.
        return [{"id": h["id"], "name": h["name"]} for h in self.store["histories"]]

    def show_history(self, history_id, contents=False, deleted=None, visible=None, details=None):
        hist = next(h for h in self.store["histories"] if h["id"] == history_id)
        if contents:
            items = [
                item
                for item in hist["contents"]
                if (deleted is None or item.get("deleted", False) == deleted)
                and (visible is None or item.get("visible", True) == visible)
            ]
            # Like Galaxy, only the detailed listing has file_size.
            if details == "all":
                return items
            return [{k: v for k, v in item.items() if k != "file_size"} for item in items]
        return {
            "id": hist["id"],
            "name": hist["name"],
            "create_time": hist["create_time"],
            "update_time": hist["update_time"],
        }


class FakeDatasetCollections:
    def __init__(self, store):
        self.store = store

    def show_dataset_collection(self, collection_id):
        return {"elements": self.store["collections"][collection_id]}


class FakeDatasets:
    def __init__(self, store):
        self.store = store

    def show_dataset(self, dataset_id, hda_ldda="hda"):
        sizes = self.store.get("dataset_sizes", {})
        return {
            "id": dataset_id,
            "file_size": sizes.get(dataset_id, 1024),
            "download_url": f"/api/datasets/{dataset_id}/display?to_ext=txt",
            "state": "ok",
        }


class FakeLibraries:
    def __init__(self, store):
        self.store = store

    def get_libraries(self):
        return [{"id": lib["id"], "name": lib["name"]} for lib in self.store["libraries"]]

    def show_dataset(self, library_id, dataset_id):
        """Like Galaxy, the only call that maps a LibraryDataset id to its LDDA."""
        lib = next(lb for lb in self.store["libraries"] if lb["id"] == library_id)
        item = next(it for it in lib["contents"] if it["id"] == dataset_id)
        return {"ldda_id": item["ldda_id"], "file_size": item["file_size"], "state": "ok"}

    def show_library(self, library_id, contents=False):
        lib = next(lb for lb in self.store["libraries"] if lb["id"] == library_id)
        if contents:
            return lib["contents"]
        return {"id": lib["id"], "name": lib["name"]}


class FakeGalaxyInstance:
    def __init__(self, store):
        self.base_url = "https://galaxy.example"
        self.key = "test-key"
        self.histories = FakeHistories(store)
        self.dataset_collections = FakeDatasetCollections(store)
        self.datasets = FakeDatasets(store)
        self.libraries = FakeLibraries(store)


def _store():
    return {
        "histories": [
            {
                "id": "hid1",
                "name": "History A",
                "create_time": "2024-01-01T00:00:00",
                "update_time": "2024-01-02T00:00:00",
                "contents": [
                    {
                        "id": "ds1",
                        "hid": 1,
                        "name": "my-uploaded-dataset",
                        "history_content_type": "dataset",
                        "file_size": 11,
                    },
                    {
                        "id": "coll1",
                        "hid": 2,
                        "name": "my result",
                        "history_content_type": "dataset_collection",
                        "collection_type": "list:paired",
                    },
                ],
            }
        ],
        "collections": {
            "coll1": [
                {
                    "element_type": "dataset_collection",
                    "element_index": 0,
                    "element_identifier": "sample1",
                    "object": {
                        "id": "subcoll1",
                        "name": "sample1",
                        "collection_type": "paired",
                    },
                },
            ],
            "subcoll1": [
                {
                    "element_type": "hda",
                    "element_index": 0,
                    "element_identifier": "forward",
                    "object": {"id": "dsF", "name": "R1"},
                },
                {
                    "element_type": "hda",
                    "element_index": 1,
                    "element_identifier": "reverse",
                    "object": {"id": "dsR", "name": "R2"},
                },
            ],
        },
        "libraries": [
            {
                "id": "lib1",
                "name": "Shared Data",
                "contents": [
                    {"id": "f_root", "type": "folder", "name": "/"},
                    {"id": "f1", "type": "folder", "name": "/genomes"},
                    {"id": "dsL1", "type": "file", "name": "/genomes/hg38.fa",
                     "ldda_id": "ldda1", "file_size": 14},
                    {"id": "dsL2", "type": "file", "name": "/reads.fastq",
                     "ldda_id": "ldda2", "file_size": 10},
                ],
            }
        ],
    }


@pytest.fixture
def fs():
    filesystem = GalaxyFileSystem(url="https://galaxy.example", api_key="test-key")
    filesystem.gi = FakeGalaxyInstance(_store())
    return filesystem


class TestRoot:
    def test_root_lists_histories_dir(self, fs):
        assert "histories" in fs.ls("/")

    def test_root_lists_libraries_dir(self, fs):
        assert "libraries" in fs.ls("/")

    def test_root_lists_both(self, fs):
        assert set(fs.ls("/")) == {"histories", "libraries"}

    def test_histories_dir_info(self, fs):
        info = fs.info("histories")
        assert info["type"] == "directory"


class TestHistories:
    def test_list_histories(self, fs):
        names = fs.ls("histories")
        assert names == ["histories/History A"]

    def test_history_info_has_dates(self, fs):
        info = fs.info("histories/History A")
        assert info["type"] == "directory"
        assert info["created"] == "2024-01-01T00:00:00"
        assert info["last_modified"] == "2024-01-02T00:00:00"
        assert info["history_id"] == "hid1"


class TestHistoryContents:
    def test_lists_what_the_history_panel_shows(self, fs):
        fs.gi.histories.store["histories"][0]["contents"] += [
            {"id": "gone", "name": "deleted-draft", "history_content_type": "dataset", "deleted": True},
            {"id": "copy", "name": "hidden-copy", "history_content_type": "dataset", "visible": False},
        ]
        entries = fs.ls("histories/History A", detail=True)
        assert [e["name"] for e in entries] == [
            "histories/History A/my-uploaded-dataset",
            "histories/History A/my result",
        ]
        assert entries[0]["size"] == 11

    def test_lists_dataset_and_collection(self, fs):
        names = fs.ls("histories/History A")
        assert "histories/History A/my-uploaded-dataset" in names
        assert "histories/History A/my result" in names

    def test_dataset_info(self, fs):
        info = fs.info("histories/History A/my-uploaded-dataset")
        assert info["type"] == "file"
        assert info["size"] == 11
        assert info["dataset_id"] == "ds1"
        assert info["hid"] == 1

    def test_collection_info(self, fs):
        info = fs.info("histories/History A/my result")
        assert info["type"] == "directory"
        assert info["collection_id"] == "coll1"


class TestNestedCollections:
    def test_list_list_paired(self, fs):
        names = fs.ls("histories/History A/my result")
        assert names == ["histories/History A/my result/sample1"]

    def test_descend_into_paired(self, fs):
        names = fs.ls("histories/History A/my result/sample1")
        assert set(names) == {
            "histories/History A/my result/sample1/forward",
            "histories/History A/my result/sample1/reverse",
        }

    def test_leaf_dataset_info(self, fs):
        info = fs.info("histories/History A/my result/sample1/forward")
        assert info["type"] == "file"
        assert info["dataset_id"] == "dsF"


class TestNumberedNames:
    def fs_numbered(self):
        f = GalaxyFileSystem(url="https://galaxy.example", api_key="test-key", show_hid_in_names=True)
        f.gi = FakeGalaxyInstance(_store())
        return f

    def test_history_contents_numbered(self):
        fs = self.fs_numbered()
        names = fs.ls("histories/History A")
        assert "histories/History A/1-my-uploaded-dataset" in names
        assert "histories/History A/2-my result" in names

    def test_collection_elements_numbered(self):
        fs = self.fs_numbered()
        names = fs.ls("histories/History A/2-my result")
        assert "histories/History A/2-my result/1-sample1" in names

    def test_paired_elements_numbered(self):
        fs = self.fs_numbered()
        names = fs.ls("histories/History A/2-my result/1-sample1")
        assert "histories/History A/2-my result/1-sample1/1-forward" in names
        assert "histories/History A/2-my result/1-sample1/2-reverse" in names

    def test_info_resolves_numbered(self):
        fs = self.fs_numbered()
        info = fs.info("histories/History A/1-my-uploaded-dataset")
        assert info["dataset_id"] == "ds1"


class TestReadOnly:
    def test_open_write_raises(self, fs):
        from galaxy_fsspec.exceptions import ReadOnlyError

        with pytest.raises(ReadOnlyError):
            fs.open("histories/History A/my-uploaded-dataset", "wb")

    def test_mkdir_raises(self, fs):
        from galaxy_fsspec.exceptions import ReadOnlyError

        with pytest.raises(ReadOnlyError):
            fs.mkdir("histories/History A/new")

    def test_rm_raises(self, fs):
        from galaxy_fsspec.exceptions import ReadOnlyError

        with pytest.raises(ReadOnlyError):
            fs.rm("histories/History A/my-uploaded-dataset")


class TestNotFound:
    def test_missing_history(self, fs):
        from galaxy_fsspec.exceptions import NotFoundError

        with pytest.raises(NotFoundError):
            fs.info("histories/Nope")

    def test_missing_dataset(self, fs):
        from galaxy_fsspec.exceptions import NotFoundError

        with pytest.raises(NotFoundError):
            fs.info("histories/History A/missing")


class TestDownloadRange:
    def test_fetch_uses_requests_206(self, fs, monkeypatch):
        import galaxy_fsspec.fs as fsmod

        captured = {}

        class FakeResp:
            status_code = 206

            def __init__(self, data):
                self._data = data

            def iter_content(self, chunk_size=8192):
                yield self._data

        def fake_get(url, headers, timeout, stream):
            captured["url"] = url
            captured["headers"] = headers
            return FakeResp(b"HELLO")

        monkeypatch.setattr(fsmod, "requests", type("R", (), {"get": staticmethod(fake_get)}))
        data = fs._download_range("ds1", 0, 5)
        assert data == b"HELLO"
        assert captured["headers"]["Range"] == "bytes=0-4"
        assert captured["headers"]["x-api-key"] == "test-key"

    def test_fetch_200_slices(self, fs, monkeypatch):
        import galaxy_fsspec.fs as fsmod

        class FakeResp:
            status_code = 200

            def __init__(self, data):
                self._data = data

            def iter_content(self, chunk_size=8192):
                yield self._data

        monkeypatch.setattr(
            fsmod, "requests", type("R", (), {"get": staticmethod(lambda *a, **k: FakeResp(b"HELLOWORLD"))})
        )
        data = fs._download_range("ds1", 2, 7)
        assert data == b"LLOWO"


class TestFileRead:
    def test_open_and_read(self, fs, monkeypatch):
        import galaxy_fsspec.fs as fsmod

        class FakeResp:
            status_code = 206

            def __init__(self, content):
                self._content = content

            def iter_content(self, chunk_size=8192):
                yield self._content

        def fake_get(url, headers, timeout, stream):
            rng = headers["Range"]
            start, end = rng[6:].split("-")
            return FakeResp(b"HELLOWORLD"[int(start) : int(end) + 1])

        monkeypatch.setattr(fsmod, "requests", type("R", (), {"get": staticmethod(fake_get)}))
        with fs.open("histories/History A/my-uploaded-dataset", "rb") as f:
            assert f.read() == b"HELLOWORLD"

    def test_a_path_starting_with_a_slash_reads_everything(self, fs, monkeypatch):
        import galaxy_fsspec.fs as fsmod

        fs.gi.datasets.store["dataset_sizes"] = {"dsF": 10}

        class FakeResp:
            status_code = 206

            def __init__(self, content):
                self._content = content

            def iter_content(self, chunk_size=8192):
                yield self._content

        def fake_get(url, headers, **kwargs):
            start, end = headers["Range"][6:].split("-")
            return FakeResp(b"R1CONTENT!"[int(start) : int(end) + 1])

        monkeypatch.setattr(fsmod, "requests", type("R", (), {"get": staticmethod(fake_get)}))
        with fs.open("/histories/History A/my result/sample1/forward", "rb") as f:
            assert f.read() == b"R1CONTENT!"

    def test_open_and_read_inside_collection(self, fs, monkeypatch):
        """A dataset leaf inside a collection is listed with size 0; opening it
        must fetch the real size via datasets.show_dataset so read() returns bytes."""
        import galaxy_fsspec.fs as fsmod

        # Real size for the forward dataset (dsF).
        fs.gi.datasets.store["dataset_sizes"] = {"dsF": 10}

        class FakeResp:
            status_code = 206

            def __init__(self, content):
                self._content = content

            def iter_content(self, chunk_size=8192):
                yield self._content

        def fake_get(url, headers, timeout, stream):
            rng = headers["Range"]
            start, end = rng[6:].split("-")
            payload = b"R1CONTENT!"[int(start) : int(end) + 1]
            return FakeResp(payload)

        monkeypatch.setattr(fsmod, "requests", type("R", (), {"get": staticmethod(fake_get)}))
        with fs.open(
            "histories/History A/my result/sample1/forward", "rb"
        ) as f:
            assert f.size == 10
            assert f.read() == b"R1CONTENT!"


class TestLibrariesRoot:
    def test_libraries_dir_info(self, fs):
        info = fs.info("libraries")
        assert info["type"] == "directory"

    def test_list_libraries(self, fs):
        names = fs.ls("libraries")
        assert "libraries/Shared Data" in names


class TestLibraryContents:
    def test_list_library_root(self, fs):
        names = fs.ls("libraries/Shared Data")
        assert "libraries/Shared Data/genomes" in names
        assert "libraries/Shared Data/reads.fastq" in names

    def test_library_folder_info(self, fs):
        info = fs.info("libraries/Shared Data/genomes")
        assert info["type"] == "directory"
        assert info["library_folder_id"] == "f1"

    def test_library_dataset_info(self, fs):
        info = fs.info("libraries/Shared Data/reads.fastq")
        assert info["type"] == "file"
        assert info["library_dataset_id"] == "dsL2"
        assert info["library_id"] == "lib1"

    def test_list_nested_folder(self, fs):
        names = fs.ls("libraries/Shared Data/genomes")
        assert "libraries/Shared Data/genomes/hg38.fa" in names

    def test_nested_dataset_info(self, fs):
        info = fs.info("libraries/Shared Data/genomes/hg38.fa")
        assert info["type"] == "file"
        assert info["library_dataset_id"] == "dsL1"


class TestLibraryFileRead:
    def test_open_and_read_library_dataset(self, fs, monkeypatch):
        """Library bytes are read from the LDDA, never by the LibraryDataset id."""
        import galaxy_fsspec.fs as fsmod

        captured = {}

        class FakeResp:
            status_code = 206

            def __init__(self, content):
                self._content = content

            def iter_content(self, chunk_size=8192):
                yield self._content

        def fake_get(url, headers, timeout, stream):
            captured["url"] = url
            captured["headers"] = headers
            rng = headers["Range"]
            start, end = rng[6:].split("-")
            return FakeResp(b"GTACGTACGTACGT"[int(start) : int(end) + 1])

        monkeypatch.setattr(fsmod, "requests", type("R", (), {"get": staticmethod(fake_get)}))
        with fs.open("libraries/Shared Data/genomes/hg38.fa", "rb") as f:
            assert f.size == 14
            assert f.read() == b"GTACGTACGTACGT"
        assert captured["url"].endswith("/api/datasets/ldda1/display?raw=true&hda_ldda=ldda")

    def test_open_and_read_root_library_dataset(self, fs, monkeypatch):
        import galaxy_fsspec.fs as fsmod

        class FakeResp:
            status_code = 206

            def __init__(self, content):
                self._content = content

            def iter_content(self, chunk_size=8192):
                yield self._content

        def fake_get(url, headers, timeout, stream):
            rng = headers["Range"]
            start, end = rng[6:].split("-")
            return FakeResp(b"ATGCATGCAT"[int(start) : int(end) + 1])

        monkeypatch.setattr(fsmod, "requests", type("R", (), {"get": staticmethod(fake_get)}))
        with fs.open("libraries/Shared Data/reads.fastq", "rb") as f:
            assert f.size == 10
            assert f.read() == b"ATGCATGCAT"


class TestLibraryNotFound:
    def test_missing_library(self, fs):
        from galaxy_fsspec.exceptions import NotFoundError

        with pytest.raises(NotFoundError):
            fs.info("libraries/Nope")

    def test_missing_library_dataset(self, fs):
        from galaxy_fsspec.exceptions import NotFoundError

        with pytest.raises(NotFoundError):
            fs.info("libraries/Shared Data/missing.txt")
