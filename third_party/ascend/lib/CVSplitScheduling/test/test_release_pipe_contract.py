from pathlib import Path


SOURCE = Path(__file__).resolve().parent.parent / "CrossScopeTransfers.cpp"


def test_merged_slot_release_orders_the_actual_reader_and_writer_pipes():
    cpp = SOURCE.read_text()
    release = cpp.split("emitMergedSlotRelease(", 1)[1].split(
        "} // namespace", 1
    )[0]

    assert "c.vecCoreAttr, c.pipeVAttr, c.pipeFixAttr" in release
    assert "c.cubeCoreAttr, c.pipeVAttr, c.pipeFixAttr" in release
    assert "c.vecCoreAttr, c.pipeMte3Attr, c.pipeMte1Attr" not in release
    assert "c.cubeCoreAttr, c.pipeMte3Attr, c.pipeMte1Attr" not in release


if __name__ == "__main__":
    test_merged_slot_release_orders_the_actual_reader_and_writer_pipes()
