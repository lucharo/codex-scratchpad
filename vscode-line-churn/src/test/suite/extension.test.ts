/**
 * Integration tests for the Line Churn extension.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Extension Test Suite', () => {
    void vscode.window.showInformationMessage('Starting extension tests.');

    test('Extension should be present', () => {
        assert.ok(
            vscode.extensions.getExtension('line-churn.line-churn'),
            'Extension should be installed'
        );
    });

    test('Commands should be registered', async () => {
        const commands = await vscode.commands.getCommands(true);

        assert.ok(
            commands.includes('lineChurn.toggle'),
            'Toggle command should be registered'
        );
        assert.ok(
            commands.includes('lineChurn.refresh'),
            'Refresh command should be registered'
        );
        assert.ok(
            commands.includes('lineChurn.showLineHistory'),
            'Show line history command should be registered'
        );
    });

    test('Configuration should have defaults', () => {
        const config = vscode.workspace.getConfiguration('lineChurn');

        assert.strictEqual(config.get('enabled'), true, 'enabled should default to true');
        assert.strictEqual(config.get('style'), 'background', 'style should default to background');
        assert.strictEqual(config.get('maxOpacity'), 0.3, 'maxOpacity should default to 0.3');
        assert.strictEqual(config.get('colorScheme'), 'heat', 'colorScheme should default to heat');
        assert.strictEqual(config.get('commitLimit'), 500, 'commitLimit should default to 500');
        assert.strictEqual(config.get('minLineLength'), 3, 'minLineLength should default to 3');
    });
});
