/**
 * Models `dw/security.py`'s validators for the path-injection query.
 *
 * Every filesystem access in this project reaches the disk through
 * `validate_path` (or a thin wrapper around it), which resolves the path
 * with `os.path.realpath` and then raises unless the result is the base
 * directory or a descendant of it; a name that is joined onto a library
 * root goes through one of the reference validators, a full-match regex
 * whose first character class precludes `..`, a leading separator and a
 * null byte. That is exactly the normalize-then-check shape the standard
 * `py/path-injection` query looks for - but it only recognizes the check as
 * a *local* barrier guard, so a validator living in another module,
 * returning the safe value rather than guarding a branch, is invisible to
 * it. Without this file the query flags every route that touches a file,
 * which buries the ones that skipped a validator.
 *
 * The trade this makes is that `dw/security.py` is now trusted by the
 * query rather than checked by it: a bug in a validator would not be
 * reported here. That file is the security layer and is reviewed as such -
 * see tests/test_security.py, which is where a validator's containment is
 * actually established.
 *
 * The model deliberately distinguishes the two ways `validate_path` is
 * called:
 *
 * - with a base directory, it is a containment check and a full barrier
 * - with `None` (or nothing) for the base, it only normalizes, so it is
 *   modeled as a `PathNormalization` and the path stays reportable until
 *   something checks it
 *
 * so a call that forgets the base directory is still a finding.
 */

private import python
private import semmle.python.Concepts
private import semmle.python.dataflow.new.DataFlow
private import semmle.python.security.dataflow.PathInjectionCustomizations

/**
 * Holds if `name` is a `dw.security` function that returns a path confined
 * to a base directory it was given.
 */
private predicate pathValidatorName(string name) {
  name =
    [
      "validate_path", "validate_workflow_path", "validate_output_path",
      "validate_prompt_path", "safe_join_path"
    ]
}

/**
 * Holds if `name` is a `dw.security` function that returns a name it has
 * matched against a pattern admitting no traversal, for joining onto a
 * library root.
 */
private predicate nameValidatorName(string name) {
  name =
    [
      "validate_workspace_name", "validate_prompt_reference", "validate_asset_reference",
      "validate_output_reference", "validate_variable_name", "validate_commit_hash"
    ]
}

/**
 * Gets the name of the function `call` invokes, for the flat and the
 * qualified spelling. Matched by name rather than by resolved definition
 * because the package imports these relatively (`from .security import
 * validate_path`), which API graphs do not track, and no other function in
 * the project carries one of these names.
 */
private string calledName(DataFlow::CallCfgNode call) {
  result = call.getFunction().asExpr().(Name).getId() or
  result = call.getFunction().asExpr().(Attribute).getName()
}

/** A call to one of the validators. */
private class ValidatorCall extends DataFlow::CallCfgNode {
  ValidatorCall() {
    pathValidatorName(calledName(this)) or
    nameValidatorName(calledName(this))
  }

  /** Gets the value this call validates. */
  DataFlow::Node getPathArg() { result = this.getArg(0) }

  /**
   * Holds if this call returns a value that cannot leave the directory it
   * will be resolved against. A name validator always does. A path
   * validator does when it was given a base directory to confine the path
   * to - `safe_join_path` joins its parts under the first one, so it
   * always confines; the others take the base as their second argument,
   * and passing `None` there asks for normalization only.
   */
  predicate confines() {
    nameValidatorName(calledName(this))
    or
    calledName(this) = "safe_join_path"
    or
    exists(DataFlow::Node base |
      base =
        [
          this.getArg(1),
          this.getArgByName(["base_dir", "workflow_dir", "prompt_dir", "output_dir"])
        ]
    |
      not base.asExpr() instanceof None
    )
  }
}

/** A value that has been validated and confined. */
private class ConfinedValue extends PathInjection::Sanitizer {
  ConfinedValue() { this.(ValidatorCall).confines() }
}

/** A path validator call given no base directory: normalization only. */
private class UnconfinedPath extends Path::PathNormalization::Range {
  UnconfinedPath() { this instanceof ValidatorCall and not this.(ValidatorCall).confines() }

  override DataFlow::Node getPathArg() { result = this.(ValidatorCall).getPathArg() }
}

/**
 * A parameter of a validator, so the query does not report the validator's
 * own `os.path.realpath` / `os.path.exists` of the value it is in the
 * middle of checking. This is the "trusted rather than checked" half of
 * the trade described at the top of this file, and it is scoped to the
 * definitions in `dw/security.py`.
 */
private class ValidatorParameter extends PathInjection::Sanitizer {
  ValidatorParameter() {
    exists(Function validator |
      pathValidatorName(validator.getName()) or nameValidatorName(validator.getName())
    |
      validator.getLocation().getFile().getRelativePath() = "dw/security.py" and
      this.asExpr() = validator.getArg(_)
    )
  }
}
