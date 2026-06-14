param(
    [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

if (-not (Test-Path ".git")) {
    throw "Run this script from the root of the cloned Git repository."
}

$groups = @{
    "scripts/train" = @(
        "run_baseline_200_clean.sh",
        "run_hfgmf_train.sh",
        "run_iswiou_train.sh",
        "run_bc_hfgmf_iswiou_train.sh"
    )
    "scripts/eval" = @(
        "run_hfgmf_val.sh",
        "run_iswiou_val.sh",
        "run_bc_hfgmf_iswiou_val.sh",
        "run_hfgmf_ab_oldstyle.sh",
        "run_iswiou_ac_oldstyle.sh",
        "run_abc_hfgmf_iswiou_oldstyle.sh",
        "run_val_native_plots.sh"
    )
    "scripts/ablation" = @(
        "run_a640_adaptive.sh",
        "run_a640_fixed.sh",
        "run_a640_native_adaptive.sh",
        "run_a640_native_fixed.sh",
        "run_a640_native_no_slice.sh",
        "run_a640_native_slice_only.sh",
        "run_a640_oldstyle_adaptive.sh",
        "run_a640_oldstyle_fixed.sh",
        "run_a640_oldstyle_no_slice.sh",
        "run_a640_slice_only.sh",
        "run_all_a640_ablation.sh",
        "run_all_a640_native_ablation.sh",
        "run_fixed_slice_640.sh",
        "run_no_slice_same_metric.sh"
    )
    "scripts/legacy" = @(
        "run_msff_fe_train.sh",
        "run_msff_fe_val.sh",
        "run_msff_fe_ab_oldstyle.sh"
    )
}

foreach ($dir in $groups.Keys) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

function Move-GitFile {
    param([string]$Source, [string]$DestinationDirectory)

    if (-not (Test-Path $Source)) {
        Write-Host "Skip missing file: $Source"
        return
    }

    $destination = Join-Path $DestinationDirectory (Split-Path $Source -Leaf)
    git ls-files --error-unmatch -- "$Source" *> $null
    if ($LASTEXITCODE -eq 0) {
        git mv -- "$Source" "$destination"
    } else {
        Move-Item -Force "$Source" "$destination"
    }
}

foreach ($entry in $groups.GetEnumerator()) {
    foreach ($file in $entry.Value) {
        Move-GitFile -Source $file -DestinationDirectory $entry.Key
    }
}

$rootBlock = @'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
'@

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

Get-ChildItem "scripts" -Recurse -Filter "*.sh" | ForEach-Object {
    $path = $_.FullName
    $content = [System.IO.File]::ReadAllText($path)
    $content = $content -replace "`r`n", "`n"
    $content = $content -replace "`r", "`n"

    # Remove the old behaviour that changed into the script's own directory.
    $content = [regex]::Replace(
        $content,
        '(?m)^\s*cd "\$\(dirname "\$0"\)"\s*\n?',
        ''
    )

    if ($content -notmatch 'REPO_ROOT=') {
        $setPattern = '(?m)^(set\s+-[^\n]+\n)'
        if ([regex]::IsMatch($content, $setPattern)) {
            $content = [regex]::Replace(
                $content,
                $setPattern,
                { param($m) $m.Groups[1].Value + "`n" + $rootBlock + "`n`n" },
                1
            )
        } elseif ($content.StartsWith("#!")) {
            $firstNewline = $content.IndexOf("`n")
            if ($firstNewline -ge 0) {
                $content = $content.Substring(0, $firstNewline + 1) +
                           "`n" + $rootBlock + "`n`n" +
                           $content.Substring($firstNewline + 1)
            }
        } else {
            $content = $rootBlock + "`n`n" + $content
        }
    }

    [System.IO.File]::WriteAllText($path, $content, $utf8NoBom)

    $relative = Resolve-Path -Relative $path
    $relative = $relative -replace '^\.\\', ''
    $relative = $relative -replace '\\', '/'
    git update-index --add --chmod=+x -- "$relative"
}

@'
*.sh text eol=lf
*.py text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
*.md text eol=lf
*.txt text eol=lf
*.json text eol=lf
*.png binary
*.jpg binary
*.jpeg binary
*.pdf binary
*.pt binary
*.pth binary
*.onnx binary
*.engine binary
'@ | Set-Content -NoNewline -Encoding utf8 ".gitattributes"

Write-Host ""
Write-Host "Shell scripts organized and path handling normalized."
Write-Host "Review the changes with:"
Write-Host "  git status --short"
Write-Host "  git diff --stat"
Write-Host "  git diff"
