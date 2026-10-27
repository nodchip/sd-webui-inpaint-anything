from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is needed to verify the browser callback')
class EditorJavascriptTests(unittest.TestCase):
    def test_wait_callback_preserves_inputs_without_appending_outputs(self):
        path = Path(__file__).resolve().parents[1] / 'javascript/ia_unified_editor.js'
        script = path.read_text(encoding='utf-8')
        check = '''
window.inpaintAnythingEditor.wait = async () => {};
(async () => {
  const cleaner = await inpaintAnything_waitForEditor(4, 'image', null, 'lama', false, []);
  const mask = await inpaintAnything_waitForEditor(1, null, 'old output');
  if (JSON.stringify(cleaner) !== JSON.stringify(['image', null, 'lama', false])) process.exit(1);
  if (JSON.stringify(mask) !== JSON.stringify([null])) process.exit(1);
})().catch(() => process.exit(1));
'''
        subprocess.run(['node', '-e', 'global.window = {}; global.onUiLoaded = () => {};\n' + script + check],
                       check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
