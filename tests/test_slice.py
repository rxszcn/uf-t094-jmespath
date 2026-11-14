import os
import sys
import json
import subprocess

from tests import unittest

import jmespath
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor


BIN_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
JP_CLI = os.path.join(BIN_DIR, 'bin', 'jp.py')

# Every slice shape that carries an explicit zero step.  The tuples are
# (expression, offending token column) so the expression context reported
# by the error can be asserted as well.
ZERO_STEP_EXPRESSIONS = [
    ('foo[8:2:0]', 8),
    ('foo[::0]', 6),
    ('foo[0:0:0]', 8),
    ('[0:0:0]', 5),
    ('[::0]', 3),
    ('foo.bar[1::0]', 11),
]

GIVEN = {
    'foo': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'bar': [0, 1, 2],
}


class TestZeroStepSliceProgrammaticAPI(unittest.TestCase):
    # Programming entry point: jmespath.search / compile / ParsedResult.
    def _assert_invalid_value(self, expression):
        with self.assertRaises(exceptions.InvalidValueError) as cm:
            jmespath.search(expression, GIVEN)
        error = cm.exception
        # Spec defined invalid-value category, not a native numeric error.
        self.assertIsInstance(error, exceptions.JMESPathError)
        self.assertNotIsInstance(error, exceptions.ParseError)
        # Backward compatibility: callers catching the base ValueError
        # (which JMESPathError derives from) must keep working.
        self.assertIsInstance(error, ValueError)
        # The error carries the offending expression context and points at
        # the zero step token, aligned with the ParseError error shape.
        self.assertEqual(error.expression, expression)
        self.assertEqual(error.token_value, 0)
        self.assertEqual(error.token_type, 'NUMBER')
        rendered = str(error)
        self.assertIn('for expression:\n"%s"' % expression, rendered)
        self.assertIn('Slice step cannot be zero', rendered)
        self.assertEqual(error.reason, 'Slice step cannot be zero')
        return error

    def test_zero_step_raises_invalid_value_error(self):
        for expression, column in ZERO_STEP_EXPRESSIONS:
            error = self._assert_invalid_value(expression)
            self.assertEqual(error.lex_position, column, expression)
            # The underline caret must land on the zero step token, one
            # column past the 0-based lex position (leading quote offset).
            underline = ' ' * (column + 1) + '^'
            self.assertIn(underline, str(error), expression)

    def test_compile_rejects_zero_step_before_evaluation(self):
        # The parser rejects the shape, so the data being searched never
        # matters; the error is raised at compile time.
        for expression, _ in ZERO_STEP_EXPRESSIONS:
            with self.assertRaises(exceptions.InvalidValueError):
                jmespath.compile(expression)

    def test_error_shape_matches_parse_error_conventions(self):
        # Same structural conventions as ParseError: position, value,
        # uppercased token type, expression attribute, and an underline.
        try:
            jmespath.search('foo[8:2:0]', GIVEN)
        except exceptions.InvalidValueError as e:
            self.assertEqual(e.lex_position, 8)
            self.assertEqual(e.token_value, 0)
            self.assertEqual(e.token_type, 'NUMBER')
            self.assertEqual(e.msg, 'Invalid jmespath expression')
            self.assertIn('^', str(e))
        else:
            self.fail('InvalidValueError was not raised')

    def test_base_class_capture_habit_kept(self):
        # Existing callers habitually capture JMESPathError / ValueError.
        for expression, _ in ZERO_STEP_EXPRESSIONS:
            with self.assertRaises(exceptions.JMESPathError):
                jmespath.search(expression, GIVEN)
            with self.assertRaises(ValueError):
                jmespath.search(expression, GIVEN)

    def test_nonzero_steps_are_unaffected(self):
        cases = {
            'foo[0:10]': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'foo[0:10:1]': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'foo[::-1]': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            'foo[::2]': [0, 2, 4, 6, 8],
            'foo[8:2:-1]': [8, 7, 6, 5, 4, 3],
            'foo[8:2:-2]': [8, 6, 4],
        }
        for expression, expected in cases.items():
            self.assertEqual(jmespath.search(expression, GIVEN), expected)


class TestZeroStepSliceEvaluationLayer(unittest.TestCase):
    # The evaluation layer must converge on the same error category even
    # when a slice node is handed in without going through the parser.
    def setUp(self):
        self.interpreter = visitor.TreeInterpreter()

    def test_visitor_rejects_zero_step(self):
        node = ast.slice(0, 5, 0)
        with self.assertRaises(exceptions.InvalidValueError) as cm:
            self.interpreter.visit(node, [1, 2, 3, 4, 5, 6])
        self.assertEqual(cm.exception.reason, 'Slice step cannot be zero')
        self.assertIsInstance(cm.exception, exceptions.JMESPathError)

    def test_visitor_keeps_normal_slices_working(self):
        self.assertEqual(
            self.interpreter.visit(ast.slice(0, 5, 2), [1, 2, 3, 4, 5, 6]),
            [1, 3, 5])
        self.assertIsNone(
            self.interpreter.visit(ast.slice(0, 5, 1), {'a': 1}))


class TestZeroStepSliceCLI(unittest.TestCase):
    # Command line entry point: the error must be distributed on the
    # invalid-value channel instead of crashing with a traceback.
    def _run_cli(self, expression):
        process = subprocess.Popen(
            [sys.executable, JP_CLI, expression],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(
            json.dumps(GIVEN).encode('utf-8'))
        # Normalize CRLF so assertions are stable across platforms.
        stderr = stderr.decode('utf-8').replace('\r\n', '\n')
        return process.returncode, stdout.decode('utf-8'), stderr

    def test_cli_reports_invalid_value_channel(self):
        for expression, _ in ZERO_STEP_EXPRESSIONS:
            returncode, stdout, stderr = self._run_cli(expression)
            self.assertEqual(returncode, 1, expression)
            # The compliance runner matches the error code on stderr.
            self.assertTrue(stderr.startswith('invalid-value: '),
                            (expression, stderr))
            # No native traceback must leak onto the defect channel.
            self.assertNotIn('Traceback', stderr, expression)
            self.assertNotIn('ValueError', stderr, expression)
            # The offending expression context is still present.
            self.assertIn('"%s"' % expression, stderr, expression)
            self.assertIn('Slice step cannot be zero', stderr, expression)
            self.assertEqual(stdout, '', expression)

    def test_cli_and_api_share_error_category(self):
        # Both entries converge on the same exception type and message.
        expression = 'foo[8:2:0]'
        try:
            jmespath.search(expression, GIVEN)
        except exceptions.InvalidValueError as e:
            api_message = str(e)
        else:
            self.fail('InvalidValueError was not raised')
        _, _, stderr = self._run_cli(expression)
        self.assertEqual(stderr, 'invalid-value: %s\n' % api_message)


if __name__ == '__main__':
    unittest.main()
