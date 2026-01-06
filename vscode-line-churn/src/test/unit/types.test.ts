/**
 * Unit tests for type definitions and error classes.
 */

import * as assert from 'assert';
import { AnalysisError, AnalysisErrorType } from '../../types';

suite('Types', () => {
    suite('AnalysisError', () => {
        test('should create error with type and message', () => {
            const error = new AnalysisError(
                AnalysisErrorType.NotGitRepository,
                'Not inside a git repository'
            );

            assert.strictEqual(error.type, AnalysisErrorType.NotGitRepository);
            assert.strictEqual(error.message, 'Not inside a git repository');
            assert.strictEqual(error.name, 'AnalysisError');
            assert.strictEqual(error.cause, undefined);
        });

        test('should create error with cause', () => {
            const cause = new Error('Original error');
            const error = new AnalysisError(
                AnalysisErrorType.GitCommandFailed,
                'Git command failed',
                cause
            );

            assert.strictEqual(error.type, AnalysisErrorType.GitCommandFailed);
            assert.strictEqual(error.cause, cause);
        });

        test('should be instance of Error', () => {
            const error = new AnalysisError(
                AnalysisErrorType.FileNotFound,
                'File not found'
            );

            assert.ok(error instanceof Error);
            assert.ok(error instanceof AnalysisError);
        });

        test('should have all error types defined', () => {
            // Verify all expected error types exist
            assert.strictEqual(AnalysisErrorType.NotGitRepository, 'NOT_GIT_REPOSITORY');
            assert.strictEqual(AnalysisErrorType.FileNotTracked, 'FILE_NOT_TRACKED');
            assert.strictEqual(AnalysisErrorType.FileNotFound, 'FILE_NOT_FOUND');
            assert.strictEqual(AnalysisErrorType.GitCommandFailed, 'GIT_COMMAND_FAILED');
            assert.strictEqual(AnalysisErrorType.ParseError, 'PARSE_ERROR');
            assert.strictEqual(AnalysisErrorType.Timeout, 'TIMEOUT');
        });
    });

    suite('AnalysisErrorType enum', () => {
        test('should have 6 error types', () => {
            const types = Object.values(AnalysisErrorType);
            assert.strictEqual(types.length, 6);
        });

        test('error types should be unique', () => {
            const types = Object.values(AnalysisErrorType);
            const uniqueTypes = new Set(types);
            assert.strictEqual(uniqueTypes.size, types.length);
        });
    });
});

suite('Result type usage', () => {
    // Test the Result pattern implementation
    type Result<T, E = Error> =
        | { success: true; value: T }
        | { success: false; error: E };

    function successResult<T>(value: T): Result<T> {
        return { success: true, value };
    }

    function failureResult<E>(error: E): Result<never, E> {
        return { success: false, error };
    }

    test('should create success result', () => {
        const result = successResult(42);

        assert.strictEqual(result.success, true);
        if (result.success) {
            assert.strictEqual(result.value, 42);
        }
    });

    test('should create failure result', () => {
        const error = new Error('Something went wrong');
        const result = failureResult(error);

        assert.strictEqual(result.success, false);
        if (!result.success) {
            assert.strictEqual(result.error, error);
        }
    });

    test('should narrow types correctly', () => {
        const successCase: Result<string> = { success: true, value: 'hello' };
        const failureCase: Result<string> = { success: false, error: new Error('fail') };

        // Type narrowing should work
        if (successCase.success) {
            assert.strictEqual(typeof successCase.value, 'string');
        }

        if (!failureCase.success) {
            assert.ok(failureCase.error instanceof Error);
        }
    });
});
