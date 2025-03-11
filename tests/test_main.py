from unittest import mock


def test_init():
    """This rather bulky test assures that our init function does everything
    it should:

        - Really calls main if we have '__name__' equals to '__main__'
        - Returns it return value as exit code
        - Does not call main() otherwise

    The final line of the code, the init() call will run at the module import
    time and, therefore, is run at test time.
    """
    from clusx import __main__ as module

    with mock.patch.object(module, "main", return_value=42):
        with mock.patch.object(module, "__name__", "__main__"):
            with mock.patch.object(module.sys, "exit") as mock_exit:
                module.init()
                assert mock_exit.call_args[0][0] == 42
