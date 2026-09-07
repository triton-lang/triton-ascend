import triton.language.extra.cann.extension as al

# CustomMacro configuration fragment, not a complete runnable example. Replace
# BITCODE_PATH with an existing bitcode file whose synchronization behavior
# matches this slot.
BITCODE_PATH = "/absolute/path/to/custom_ops.bc"


@al.register_custom_op
class sync_event_slot_example_macro:
    core = al.CORE.VECTOR
    pipe = (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V)
    mode = al.MODE.SIMD
    symbol = "sync_event_slot_example_func"
    bitcode = BITCODE_PATH
    sync_event_slots = [
        al.SyncEventSlot(
            set_pipe=al.PIPE.PIPE_MTE2,
            wait_pipe=al.PIPE.PIPE_MTE1,
            # The slot pipelines describe synchronization inside the device
            # function and may differ from the macro input/output pipelines.
            sync=al.SYNC_HINT.WAIT,
            event=al.EVENT_ID.EVENT_ID1,
        )
    ]
