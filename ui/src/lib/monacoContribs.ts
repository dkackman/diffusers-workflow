/** Monaco's editor contributions: every `./editor/*`, `./features/*` and
 * codicon CSS import `monaco-editor/esm/vs/index.js` (the whole package)
 * makes, in its order, and none of its language definitions, other
 * language services or LSP client - `monaco.ts` adds the JSON language
 * alone. Find, folding, hover, completion, the context menu, the command
 * palette and the codicon font all come from here.
 * `monacoContribs.test.ts` pins this list to the installed package, so a
 * Monaco upgrade that adds or drops a contribution fails until it is
 * mirrored here.
 *
 * The package's exports map every `monaco-editor/<path>` onto
 * `esm/vs/<path>.js`, which no stylesheet is, so the two codicon CSS files
 * are imported by their path under node_modules - the same files the
 * contributions themselves import. */
import 'monaco-editor/editor/contrib/anchorSelect/browser/anchorSelect.js'
import 'monaco-editor/editor/contrib/bracketMatching/browser/bracketMatching.js'
import 'monaco-editor/editor/contrib/caretOperations/browser/transpose.js'
import 'monaco-editor/editor/contrib/clipboard/browser/clipboard.js'
import 'monaco-editor/editor/contrib/codeAction/browser/codeActionContributions.js'
import 'monaco-editor/editor/browser/widget/codeEditor/codeEditorWidget.js'
import 'monaco-editor/editor/contrib/codelens/browser/codelensController.js'
import '../../node_modules/monaco-editor/esm/vs/base/browser/ui/codicons/codicon/codicon.css'
import 'monaco-editor/editor/contrib/colorPicker/browser/colorPickerContribution.js'
import 'monaco-editor/editor/contrib/comment/browser/comment.js'
import 'monaco-editor/editor/contrib/contextmenu/browser/contextmenu.js'
import 'monaco-editor/editor/contrib/cursorUndo/browser/cursorUndo.js'
import 'monaco-editor/editor/browser/widget/diffEditor/diffEditor.contribution.js'
import 'monaco-editor/editor/contrib/diffEditorBreadcrumbs/browser/contribution.js'
import 'monaco-editor/editor/contrib/dnd/browser/dnd.js'
import 'monaco-editor/editor/contrib/documentSymbols/browser/documentSymbols.js'
import 'monaco-editor/editor/contrib/dropOrPasteInto/browser/dropIntoEditorContribution.js'
import 'monaco-editor/features/find/register.js'
import 'monaco-editor/editor/contrib/floatingMenu/browser/floatingMenu.contribution.js'
import 'monaco-editor/editor/contrib/folding/browser/folding.js'
import 'monaco-editor/editor/contrib/fontZoom/browser/fontZoom.js'
import 'monaco-editor/editor/contrib/format/browser/formatActions.js'
import 'monaco-editor/editor/contrib/gotoError/browser/gotoError.js'
import 'monaco-editor/editor/standalone/browser/quickAccess/standaloneGotoLineQuickAccess.js'
import 'monaco-editor/editor/contrib/gotoSymbol/browser/link/goToDefinitionAtPosition.js'
import 'monaco-editor/editor/contrib/gpu/browser/gpuActions.js'
import 'monaco-editor/editor/contrib/hover/browser/hoverContribution.js'
import 'monaco-editor/editor/contrib/indentation/browser/indentation.js'
import 'monaco-editor/editor/contrib/inlayHints/browser/inlayHintsContribution.js'
import 'monaco-editor/editor/contrib/inlineCompletions/browser/inlineCompletions.contribution.js'
import 'monaco-editor/editor/contrib/inlineProgress/browser/inlineProgress.js'
import 'monaco-editor/editor/contrib/inPlaceReplace/browser/inPlaceReplace.js'
import 'monaco-editor/editor/contrib/insertFinalNewLine/browser/insertFinalNewLine.js'
import 'monaco-editor/editor/standalone/browser/inspectTokens/inspectTokens.js'
import 'monaco-editor/editor/standalone/browser/iPadShowKeyboard/iPadShowKeyboard.js'
import 'monaco-editor/editor/contrib/lineSelection/browser/lineSelection.js'
import 'monaco-editor/editor/contrib/linesOperations/browser/linesOperations.js'
import 'monaco-editor/editor/contrib/linkedEditing/browser/linkedEditing.js'
import 'monaco-editor/editor/contrib/links/browser/links.js'
import 'monaco-editor/editor/contrib/longLinesHelper/browser/longLinesHelper.js'
import 'monaco-editor/editor/contrib/middleScroll/browser/middleScroll.contribution.js'
import 'monaco-editor/editor/contrib/multicursor/browser/multicursor.js'
import 'monaco-editor/editor/contrib/parameterHints/browser/parameterHints.js'
import 'monaco-editor/editor/contrib/placeholderText/browser/placeholderText.contribution.js'
import 'monaco-editor/editor/standalone/browser/quickAccess/standaloneCommandsQuickAccess.js'
import 'monaco-editor/editor/standalone/browser/quickAccess/standaloneHelpQuickAccess.js'
import 'monaco-editor/editor/standalone/browser/quickAccess/standaloneGotoSymbolQuickAccess.js'
import 'monaco-editor/editor/contrib/readOnlyMessage/browser/contribution.js'
import 'monaco-editor/editor/standalone/browser/referenceSearch/standaloneReferenceSearch.js'
import 'monaco-editor/editor/contrib/rename/browser/rename.js'
import 'monaco-editor/editor/contrib/sectionHeaders/browser/sectionHeaders.js'
import 'monaco-editor/editor/contrib/semanticTokens/browser/viewportSemanticTokens.js'
import 'monaco-editor/editor/contrib/smartSelect/browser/smartSelect.js'
import 'monaco-editor/editor/contrib/snippet/browser/snippetController2.js'
import 'monaco-editor/editor/contrib/stickyScroll/browser/stickyScrollContribution.js'
import 'monaco-editor/editor/contrib/suggest/browser/suggestInlineCompletions.js'
import 'monaco-editor/editor/standalone/browser/toggleHighContrast/toggleHighContrast.js'
import 'monaco-editor/editor/contrib/toggleTabFocusMode/browser/toggleTabFocusMode.js'
import 'monaco-editor/editor/contrib/tokenization/browser/tokenization.js'
import 'monaco-editor/editor/contrib/unicodeHighlighter/browser/unicodeHighlighter.js'
import 'monaco-editor/editor/contrib/unusualLineTerminators/browser/unusualLineTerminators.js'
import 'monaco-editor/editor/contrib/wordHighlighter/browser/wordHighlighter.js'
import 'monaco-editor/editor/contrib/wordOperations/browser/wordOperations.js'
import 'monaco-editor/editor/contrib/wordPartOperations/browser/wordPartOperations.js'
import 'monaco-editor/editor/browser/coreCommands.js'
import 'monaco-editor/editor/contrib/caretOperations/browser/caretOperations.js'
import 'monaco-editor/editor/contrib/dropOrPasteInto/browser/copyPasteContribution.js'
import 'monaco-editor/editor/contrib/find/browser/findController.js'
import 'monaco-editor/editor/contrib/gotoSymbol/browser/goToCommands.js'
import 'monaco-editor/editor/contrib/gotoError/browser/markerSelectionStatus.js'
import 'monaco-editor/editor/contrib/semanticTokens/browser/documentSemanticTokens.js'
import 'monaco-editor/editor/contrib/suggest/browser/suggestController.js'
import 'monaco-editor/editor/common/standaloneStrings.js'
import '../../node_modules/monaco-editor/esm/vs/base/browser/ui/codicons/codicon/codicon-modifiers.css'

/** The specifiers above, in order, for the test */
export const CONTRIBS: readonly string[] = [
  'monaco-editor/editor/contrib/anchorSelect/browser/anchorSelect.js',
  'monaco-editor/editor/contrib/bracketMatching/browser/bracketMatching.js',
  'monaco-editor/editor/contrib/caretOperations/browser/transpose.js',
  'monaco-editor/editor/contrib/clipboard/browser/clipboard.js',
  'monaco-editor/editor/contrib/codeAction/browser/codeActionContributions.js',
  'monaco-editor/editor/browser/widget/codeEditor/codeEditorWidget.js',
  'monaco-editor/editor/contrib/codelens/browser/codelensController.js',
  'monaco-editor/base/browser/ui/codicons/codicon/codicon.css',
  'monaco-editor/editor/contrib/colorPicker/browser/colorPickerContribution.js',
  'monaco-editor/editor/contrib/comment/browser/comment.js',
  'monaco-editor/editor/contrib/contextmenu/browser/contextmenu.js',
  'monaco-editor/editor/contrib/cursorUndo/browser/cursorUndo.js',
  'monaco-editor/editor/browser/widget/diffEditor/diffEditor.contribution.js',
  'monaco-editor/editor/contrib/diffEditorBreadcrumbs/browser/contribution.js',
  'monaco-editor/editor/contrib/dnd/browser/dnd.js',
  'monaco-editor/editor/contrib/documentSymbols/browser/documentSymbols.js',
  'monaco-editor/editor/contrib/dropOrPasteInto/browser/dropIntoEditorContribution.js',
  'monaco-editor/features/find/register.js',
  'monaco-editor/editor/contrib/floatingMenu/browser/floatingMenu.contribution.js',
  'monaco-editor/editor/contrib/folding/browser/folding.js',
  'monaco-editor/editor/contrib/fontZoom/browser/fontZoom.js',
  'monaco-editor/editor/contrib/format/browser/formatActions.js',
  'monaco-editor/editor/contrib/gotoError/browser/gotoError.js',
  'monaco-editor/editor/standalone/browser/quickAccess/standaloneGotoLineQuickAccess.js',
  'monaco-editor/editor/contrib/gotoSymbol/browser/link/goToDefinitionAtPosition.js',
  'monaco-editor/editor/contrib/gpu/browser/gpuActions.js',
  'monaco-editor/editor/contrib/hover/browser/hoverContribution.js',
  'monaco-editor/editor/contrib/indentation/browser/indentation.js',
  'monaco-editor/editor/contrib/inlayHints/browser/inlayHintsContribution.js',
  'monaco-editor/editor/contrib/inlineCompletions/browser/inlineCompletions.contribution.js',
  'monaco-editor/editor/contrib/inlineProgress/browser/inlineProgress.js',
  'monaco-editor/editor/contrib/inPlaceReplace/browser/inPlaceReplace.js',
  'monaco-editor/editor/contrib/insertFinalNewLine/browser/insertFinalNewLine.js',
  'monaco-editor/editor/standalone/browser/inspectTokens/inspectTokens.js',
  'monaco-editor/editor/standalone/browser/iPadShowKeyboard/iPadShowKeyboard.js',
  'monaco-editor/editor/contrib/lineSelection/browser/lineSelection.js',
  'monaco-editor/editor/contrib/linesOperations/browser/linesOperations.js',
  'monaco-editor/editor/contrib/linkedEditing/browser/linkedEditing.js',
  'monaco-editor/editor/contrib/links/browser/links.js',
  'monaco-editor/editor/contrib/longLinesHelper/browser/longLinesHelper.js',
  'monaco-editor/editor/contrib/middleScroll/browser/middleScroll.contribution.js',
  'monaco-editor/editor/contrib/multicursor/browser/multicursor.js',
  'monaco-editor/editor/contrib/parameterHints/browser/parameterHints.js',
  'monaco-editor/editor/contrib/placeholderText/browser/placeholderText.contribution.js',
  'monaco-editor/editor/standalone/browser/quickAccess/standaloneCommandsQuickAccess.js',
  'monaco-editor/editor/standalone/browser/quickAccess/standaloneHelpQuickAccess.js',
  'monaco-editor/editor/standalone/browser/quickAccess/standaloneGotoSymbolQuickAccess.js',
  'monaco-editor/editor/contrib/readOnlyMessage/browser/contribution.js',
  'monaco-editor/editor/standalone/browser/referenceSearch/standaloneReferenceSearch.js',
  'monaco-editor/editor/contrib/rename/browser/rename.js',
  'monaco-editor/editor/contrib/sectionHeaders/browser/sectionHeaders.js',
  'monaco-editor/editor/contrib/semanticTokens/browser/viewportSemanticTokens.js',
  'monaco-editor/editor/contrib/smartSelect/browser/smartSelect.js',
  'monaco-editor/editor/contrib/snippet/browser/snippetController2.js',
  'monaco-editor/editor/contrib/stickyScroll/browser/stickyScrollContribution.js',
  'monaco-editor/editor/contrib/suggest/browser/suggestInlineCompletions.js',
  'monaco-editor/editor/standalone/browser/toggleHighContrast/toggleHighContrast.js',
  'monaco-editor/editor/contrib/toggleTabFocusMode/browser/toggleTabFocusMode.js',
  'monaco-editor/editor/contrib/tokenization/browser/tokenization.js',
  'monaco-editor/editor/contrib/unicodeHighlighter/browser/unicodeHighlighter.js',
  'monaco-editor/editor/contrib/unusualLineTerminators/browser/unusualLineTerminators.js',
  'monaco-editor/editor/contrib/wordHighlighter/browser/wordHighlighter.js',
  'monaco-editor/editor/contrib/wordOperations/browser/wordOperations.js',
  'monaco-editor/editor/contrib/wordPartOperations/browser/wordPartOperations.js',
  'monaco-editor/editor/browser/coreCommands.js',
  'monaco-editor/editor/contrib/caretOperations/browser/caretOperations.js',
  'monaco-editor/editor/contrib/dropOrPasteInto/browser/copyPasteContribution.js',
  'monaco-editor/editor/contrib/find/browser/findController.js',
  'monaco-editor/editor/contrib/gotoSymbol/browser/goToCommands.js',
  'monaco-editor/editor/contrib/gotoError/browser/markerSelectionStatus.js',
  'monaco-editor/editor/contrib/semanticTokens/browser/documentSemanticTokens.js',
  'monaco-editor/editor/contrib/suggest/browser/suggestController.js',
  'monaco-editor/editor/common/standaloneStrings.js',
  'monaco-editor/base/browser/ui/codicons/codicon/codicon-modifiers.css',
]
