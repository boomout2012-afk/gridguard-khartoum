param([string]$OutputPath = "")

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $projectRoot "output\presentation\gridguard-khartoum-defense.pptx"
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$renderDirectory = Join-Path $projectRoot "tmp\presentation\rendered"
$overviewImage = Join-Path $projectRoot "docs\assets\screenshots\overview.png"
$liveImage = Join-Path $projectRoot "docs\assets\screenshots\live.png"
$pairedEvidencePath = Join-Path $projectRoot "data\evidence\paired-benchmark.json"
$robustnessEvidencePath = Join-Path $projectRoot "data\evidence\robustness-supplement.json"
$trainingProvenancePath = Join-Path $projectRoot "data\model\training-provenance.yaml"
$modelManifestPath = Join-Path $projectRoot "data\model\manifest.json"
$trainingToolPath = Join-Path $projectRoot "tools\training_provenance.py"
$expectedPairedSha256 = "6052e9e47d92e82a0f1db0d3643a2e9be6edc5f3e9fc615b0b185964b54c1677"
$expectedRobustnessSha256 = "c89e04b1bdca371e1728f6b4cb19f4e899261ab805c2e400c15e835eaf75a30b"
$expectedTrainingSha256 = "2cfa158a4b26596fd1f155c57fa79a88dc1c41ce8df946792c559184a815128a"

foreach ($path in @(
        $overviewImage,
        $liveImage,
        $pairedEvidencePath,
        $robustnessEvidencePath,
        $trainingProvenancePath,
        $modelManifestPath,
        $trainingToolPath
    )) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing presentation input: $path"
    }
}

function Assert-FileHash {
    param([string]$Path, [string]$Expected, [string]$Label)

    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -cne $Expected) {
        throw "$Label SHA-256 mismatch: expected $Expected, got $actual"
    }
}

Assert-FileHash $pairedEvidencePath $expectedPairedSha256 "Accepted paired evidence"
Assert-FileHash $robustnessEvidencePath $expectedRobustnessSha256 "Robustness supplement"
Assert-FileHash $trainingProvenancePath $expectedTrainingSha256 "Training provenance"

$pairedEvidence = [System.IO.File]::ReadAllText($pairedEvidencePath) | ConvertFrom-Json
$robustnessEvidence = [System.IO.File]::ReadAllText($robustnessEvidencePath) | ConvertFrom-Json
if ($pairedEvidence.schema_version -ne 2 -or $pairedEvidence.evidence_role -cne "accepted_reproduction") {
    throw "Paired evidence must remain schema version 2 accepted reproduction context"
}
if (
    $robustnessEvidence.schema.name -cne "gridguard.robustness_supplement" -or
    $robustnessEvidence.schema.version -ne 1 -or
    $robustnessEvidence.evidence_role -cne "untouched_confirmatory" -or
    $robustnessEvidence.provenance.artifact_id -cne "gridguard-robustness-supplement-v1"
) {
    throw "Robustness supplement identity does not match the reviewed contract"
}
foreach ($controllers in @($pairedEvidence.protocol.controllers, $robustnessEvidence.protocol.controllers)) {
    if ((@($controllers) | Sort-Object) -join "|" -cne "Fixed Rotation|PPO") {
        throw "Evidence must contain exactly Fixed Rotation and PPO"
    }
}

$primaryMetricKeys = @("critical_outage_locality_hours", "fairness", "utilization")
$baseMetricKeys = @(
    $robustnessEvidence.base_evaluation.paired_inference.PSObject.Properties.Name |
        Sort-Object
)
if (($baseMetricKeys -join "|") -cne (($primaryMetricKeys | Sort-Object) -join "|")) {
    throw "Untouched evidence must contain exactly the three primary outcomes"
}
foreach ($metricKey in $primaryMetricKeys) {
    if (
        $robustnessEvidence.metric_definitions.$metricKey.role -cne "primary" -or
        $pairedEvidence.metric_definitions.$metricKey.role -cne "primary"
    ) {
        throw "$metricKey must remain primary"
    }
}
if (
    $pairedEvidence.metric_definitions.episode_reward.role -cne "diagnostic" -or
    $pairedEvidence.metric_definitions.episode_reward.label -cne "Training-objective score (diagnostic)"
) {
    throw "Episode reward must remain a training-objective diagnostic"
}

$trainingArguments = @(
    "run", "--project", $projectRoot, "--extra", "docs", "python", $trainingToolPath,
    "--path", $trainingProvenancePath, "--model-manifest", $modelManifestPath, "--summary-json"
)
$trainingOutput = & uv @trainingArguments
if ($LASTEXITCODE -ne 0) {
    throw "Strict training provenance validation failed with exit code $LASTEXITCODE"
}
$trainingSummary = ($trainingOutput -join [Environment]::NewLine) | ConvertFrom-Json
if (
    $trainingSummary.controller -cne "PPO" -or
    $trainingSummary.training_provenance.sha256 -cne $expectedTrainingSha256
) {
    throw "Training summary does not bind the expected PPO provenance"
}

$robustnessProtocol = $robustnessEvidence.protocol
$baseDrawCount = [int]$robustnessProtocol.independent_base_draw_count
$shortageCount = @($robustnessProtocol.shortage_levels).Count
$matchedScenarioCount = [int]$robustnessProtocol.matched_scenario_count
$controllerEvaluationCount = [int]$robustnessProtocol.controller_evaluation_count
$controllerCount = @($robustnessProtocol.controllers).Count
$sensitivityParameterizationCount = [int]$robustnessProtocol.sensitivity_parameterization_count
$sensitivityVariants = @($robustnessEvidence.sensitivity.variants)
if (
    $baseDrawCount -ne 320 -or
    $shortageCount -ne 3 -or
    $matchedScenarioCount -ne 960 -or
    $controllerEvaluationCount -ne 1920 -or
    $baseDrawCount * $shortageCount -ne $matchedScenarioCount -or
    $matchedScenarioCount * $controllerCount -ne $controllerEvaluationCount
) {
    throw "Robustness protocol no longer matches the locked 320/960/1920 design"
}
if ($sensitivityParameterizationCount -ne 25 -or $sensitivityVariants.Count -ne 24) {
    throw "Sensitivity design must contain one base plus 24 variants"
}
if (
    [int]$pairedEvidence.protocol.independent_base_draw_count -ne 96 -or
    [int]$pairedEvidence.protocol.matched_scenario_count -ne 288 -or
    [int]$pairedEvidence.protocol.controller_evaluation_count -ne 576
) {
    throw "Accepted reproduction no longer matches the 96/288/576 context"
}
$seedOverlap = @(
    @($pairedEvidence.protocol.seeds) |
        Where-Object { @($robustnessProtocol.seeds) -contains $_ }
)
if ($seedOverlap.Count -ne 0) {
    throw "Accepted and untouched evaluation seeds must remain disjoint"
}

$robustnessBase = $robustnessEvidence.base_evaluation
$fixed = $robustnessBase.controller_means.'Fixed Rotation'
$ppo = $robustnessBase.controller_means.PPO
$inference = $robustnessBase.paired_inference
$powerBalanceRate = [double]$robustnessBase.power_balance.maximum_violation_rate
$acceptedFixed = $pairedEvidence.aggregate.'Fixed Rotation'
$acceptedPpo = $pairedEvidence.aggregate.PPO
$acceptedInference = $pairedEvidence.paired_inference
$forecast = $robustnessEvidence.forecast_evaluation
$persistence = $forecast.methods.persistence_1h.pooled
$trailingMean = $forecast.methods.trailing_mean_3h.pooled

function Count-SensitivityClass {
    param([string]$Metric, [string]$Classification, [string]$Layer = "paired_inference")

    return @(
        $sensitivityVariants |
            Where-Object { $_.$Layer.$Metric.result_classification -ceq $Classification }
    ).Count
}

$sensitivityFavorable = @{}
foreach ($metricKey in $primaryMetricKeys) {
    $sensitivityFavorable[$metricKey] = Count-SensitivityClass $metricKey "favorable"
}
$outageDidAdverse = Count-SensitivityClass "critical_outage_locality_hours" "adverse" "difference_in_differences"
$outageDidFavorable = Count-SensitivityClass "critical_outage_locality_hours" "favorable" "difference_in_differences"
$outageDidAmbiguous = Count-SensitivityClass "critical_outage_locality_hours" "ambiguous" "difference_in_differences"

$script:SlideWidth = 960.0
$script:SlideHeight = 540.0
$script:Navy = "071923"
$script:Slate = "102A3B"
$script:SoftSlate = "1A394A"
$script:Teal = "3FC6AC"
$script:Gold = "E0B957"
$script:OffWhite = "F3F0E8"
$script:Ink = "13232E"
$script:Muted = "83939D"
$script:Grid = "365568"
$script:Risk = "A15E40"

function ConvertTo-Rgb {
    param([string]$Hex)

    $clean = $Hex.TrimStart("#")
    if ($clean.Length -ne 6) {
        throw "Expected a six-digit hex color"
    }
    $red = [Convert]::ToInt32($clean.Substring(0, 2), 16)
    $green = [Convert]::ToInt32($clean.Substring(2, 2), 16)
    $blue = [Convert]::ToInt32($clean.Substring(4, 2), 16)
    return $red + (256 * $green) + (65536 * $blue)
}

function Set-SlideBackground {
    param($Slide, [string]$Color)

    $Slide.FollowMasterBackground = 0
    $Slide.Background.Fill.Visible = -1
    $Slide.Background.Fill.Solid()
    $Slide.Background.Fill.ForeColor.RGB = ConvertTo-Rgb $Color
}

function Add-Text {
    param(
        $Slide, [string]$Text, [double]$Left, [double]$Top, [double]$Width,
        [double]$Height, [double]$FontSize, [string]$Color,
        [bool]$Bold = $false, [int]$Alignment = 1, [string]$FontName = "Aptos"
    )

    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.Fill.Visible = 0
    $shape.Line.Visible = 0
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.AutoSize = 0
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = $FontName
    $shape.TextFrame.TextRange.Font.Size = $FontSize
    $shape.TextFrame.TextRange.Font.Bold = if ($Bold) { -1 } else { 0 }
    $shape.TextFrame.TextRange.Font.Color.RGB = ConvertTo-Rgb $Color
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Alignment
    $shape.Width = $Width
    $shape.Height = $Height
    return $shape
}

function Add-Box {
    param(
        $Slide, [double]$Left, [double]$Top, [double]$Width, [double]$Height,
        [string]$FillColor, [string]$LineColor = "", [int]$Geometry = 5,
        [double]$Transparency = 0
    )

    $shape = $Slide.Shapes.AddShape($Geometry, $Left, $Top, $Width, $Height)
    $shape.Fill.Visible = -1
    $shape.Fill.Solid()
    $shape.Fill.ForeColor.RGB = ConvertTo-Rgb $FillColor
    $shape.Fill.Transparency = $Transparency
    if ([string]::IsNullOrWhiteSpace($LineColor)) {
        $shape.Line.Visible = 0
    }
    else {
        $shape.Line.Visible = -1
        $shape.Line.ForeColor.RGB = ConvertTo-Rgb $LineColor
        $shape.Line.Weight = 1.2
    }
    return $shape
}

function Add-Line {
    param(
        $Slide, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2,
        [string]$Color, [double]$Weight = 1.5, [bool]$Arrow = $false
    )

    $line = $Slide.Shapes.AddLine($X1, $Y1, $X2, $Y2)
    $line.Line.ForeColor.RGB = ConvertTo-Rgb $Color
    $line.Line.Weight = $Weight
    if ($Arrow) {
        $line.Line.EndArrowheadStyle = 3
    }
    return $line
}

function Add-Picture {
    param(
        $Slide, [string]$Path, [double]$Left, [double]$Top,
        [double]$Width, [double]$Height, [string]$AltText
    )

    $picture = $Slide.Shapes.AddPicture($Path, 0, -1, $Left, $Top, $Width, $Height)
    $picture.AlternativeText = $AltText
    return $picture
}

function Add-Chrome {
    param($Slide, [int]$Number, [bool]$Light = $false)

    $muted = if ($Light) { "60727D" } else { $script:Muted }
    Add-Text $Slide "GRIDGUARD KHARTOUM" 48 18 300 22 16 $muted $true | Out-Null
    Add-Line $Slide 48 42 912 42 $(if ($Light) { "CFD8DC" } else { $script:Grid }) 0.8 | Out-Null
    Add-Text $Slide ([string]$Number) 878 503 34 22 16 $muted $false 3 | Out-Null
    Add-Text $Slide "OFFLINE GPU RESEARCH SIMULATION" 48 503 320 22 16 $muted | Out-Null
}

function Add-SlideTitle {
    param($Slide, [string]$Text, [bool]$Light = $false)

    $color = if ($Light) { $script:Ink } else { $script:OffWhite }
    Add-Text $Slide $Text 48 58 864 88 35 $color $true | Out-Null
}

function Add-BulletList {
    param(
        $Slide, [string[]]$Items, [double]$Left, [double]$Top, [double]$Width,
        [double]$FontSize = 20, [string]$Color = "F3F0E8",
        [double]$Gap = 50, [string]$Accent = "E0B957"
    )

    for ($index = 0; $index -lt $Items.Count; $index++) {
        $itemTop = $Top + ($index * $Gap)
        Add-Line $Slide $Left ($itemTop + 12) ($Left + 16) ($itemTop + 12) $Accent 3 | Out-Null
        Add-Text $Slide $Items[$index] ($Left + 28) $itemTop ($Width - 28) ($Gap - 3) $FontSize $Color | Out-Null
    }
}

function New-Notes {
    param([string]$Narrative, [string[]]$Sources)

    $lines = @($Narrative, "", "[Sources]")
    foreach ($source in $Sources) {
        $lines += "- $source"
    }
    $lines += "[/Sources]"
    return ($lines -join [Environment]::NewLine)
}

function Add-Notes {
    param($Slide, [string]$Notes)

    $notesPage = $Slide.NotesPage
    $body = $null
    for ($index = 1; $index -le $notesPage.Shapes.Count; $index++) {
        $candidate = $notesPage.Shapes.Item($index)
        if ($candidate.HasTextFrame -eq -1 -and $candidate.Name -match "Text Placeholder|Body") {
            $body = $candidate
            break
        }
    }
    if ($null -eq $body) {
        $body = $notesPage.Shapes.Placeholders.Item(2)
    }
    $body.TextFrame.TextRange.Text = $Notes
}

function Assert-PresentationLayout {
    param($Presentation)

    $tolerance = 1.0
    for ($slideIndex = 1; $slideIndex -le $Presentation.Slides.Count; $slideIndex++) {
        $slide = $Presentation.Slides.Item($slideIndex)
        for ($shapeIndex = 1; $shapeIndex -le $slide.Shapes.Count; $shapeIndex++) {
            $shape = $slide.Shapes.Item($shapeIndex)
            $right = [double]$shape.Left + [double]$shape.Width
            $bottom = [double]$shape.Top + [double]$shape.Height
            if (
                [double]$shape.Left -lt -$tolerance -or
                [double]$shape.Top -lt -$tolerance -or
                $right -gt ($script:SlideWidth + $tolerance) -or
                $bottom -gt ($script:SlideHeight + $tolerance)
            ) {
                throw "Slide $slideIndex shape '$($shape.Name)' exceeds the slide canvas"
            }
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                $range = $shape.TextFrame.TextRange
                $boundWidth = [double]$range.BoundWidth
                $boundHeight = [double]$range.BoundHeight
                if (
                    $boundWidth -gt ([double]$shape.Width + $tolerance) -or
                    $boundHeight -gt ([double]$shape.Height + $tolerance)
                ) {
                    throw (
                        "Slide $slideIndex text overflow in '$($shape.Name)': " +
                        "bound ${boundWidth}x${boundHeight}, frame $($shape.Width)x$($shape.Height)"
                    )
                }
            }
        }
    }
}

function Format-OneDecimal {
    param([double]$Value, [bool]$Signed = $false)

    $prefix = if ($Signed -and $Value -gt 0) { "+" } else { "" }
    return $prefix + $Value.ToString("0.0", [Globalization.CultureInfo]::InvariantCulture)
}

function Format-TwoDecimals {
    param([double]$Value, [bool]$Signed = $false)

    $prefix = if ($Signed -and $Value -gt 0) { "+" } else { "" }
    return $prefix + $Value.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture)
}

function Format-Percent {
    param([double]$Value)

    return (($Value * 100).ToString("0.00", [Globalization.CultureInfo]::InvariantCulture) + "%")
}

function Format-Points {
    param([double]$Value)

    $scaled = $Value * 100
    $prefix = if ($scaled -gt 0) { "+" } else { "" }
    return ($prefix + $scaled.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture) + " pp")
}

function Format-OutageInterval {
    param([double]$Lower, [double]$Upper)

    return "[{0}, {1}] locality-hours" -f (Format-OneDecimal $Lower), (Format-OneDecimal $Upper)
}

function Format-PointsInterval {
    param([double]$Lower, [double]$Upper)

    return "[{0}, {1}] pp" -f (Format-TwoDecimals ($Lower * 100)), (Format-TwoDecimals ($Upper * 100))
}

function Add-OutcomeRow {
    param(
        $Slide, [double]$Top, [string]$Metric, [string]$FixedValue,
        [string]$PpoValue, [string]$Delta, [string]$Interval,
        [bool]$Shaded = $false
    )

    if ($Shaded) {
        Add-Box $Slide 54 ($Top - 7) 852 51 "E7ECEE" "" 1 | Out-Null
    }
    Add-Text $Slide $Metric 64 $Top 226 44 17 $script:Ink $true | Out-Null
    Add-Text $Slide $FixedValue 300 $Top 126 44 18 "60727D" $true | Out-Null
    Add-Text $Slide $PpoValue 434 $Top 126 44 18 $script:Teal $true | Out-Null
    Add-Text $Slide $Delta 568 $Top 112 44 18 $script:Teal $true | Out-Null
    Add-Text $Slide $Interval 688 $Top 204 44 16 $script:Ink | Out-Null
}

function Add-StatCard {
    param(
        $Slide, [double]$Left, [double]$Top, [double]$Width, [double]$Height,
        [string]$Value, [string]$Label, [string]$FillColor,
        [string]$ValueColor = "F3F0E8"
    )

    Add-Box $Slide $Left $Top $Width $Height $FillColor "" 5 | Out-Null
    Add-Text $Slide $Value ($Left + 14) ($Top + 12) ($Width - 28) 58 23 $ValueColor $true 2 | Out-Null
    Add-Text $Slide $Label ($Left + 14) ($Top + 76) ($Width - 28) ($Height - 84) 16 $script:OffWhite $true 2 | Out-Null
}

$outageDelta = (Format-OneDecimal ([double]$inference.critical_outage_locality_hours.mean_delta) $true) + " locality-hours"
$outageInterval = Format-OutageInterval ([double]$inference.critical_outage_locality_hours.ci95_lower) ([double]$inference.critical_outage_locality_hours.ci95_upper)
$fairnessDelta = Format-Points ([double]$inference.fairness.mean_delta)
$fairnessInterval = Format-PointsInterval ([double]$inference.fairness.ci95_lower) ([double]$inference.fairness.ci95_upper)
$utilizationDelta = Format-Points ([double]$inference.utilization.mean_delta)
$utilizationInterval = Format-PointsInterval ([double]$inference.utilization.ci95_lower) ([double]$inference.utilization.ci95_upper)
$powerBalanceLabel = Format-Percent $powerBalanceRate

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null
New-Item -ItemType Directory -Force -Path $renderDirectory | Out-Null

$powerPoint = $null
$presentation = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $presentation = $powerPoint.Presentations.Add()
    $presentation.PageSetup.SlideWidth = $script:SlideWidth
    $presentation.PageSetup.SlideHeight = $script:SlideHeight

    # 1 - title
    $slide = $presentation.Slides.Add(1, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Text $slide "GRADUATION PROJECT DEFENSE" 60 58 420 24 16 $script:Teal $true | Out-Null
    Add-Text $slide "GridGuard Khartoum" 60 128 780 70 54 $script:OffWhite $true | Out-Null
    Add-Text $slide "PPO-assisted load shedding and energy management for developing countries" 60 212 760 68 26 $script:Gold | Out-Null
    Add-Line $slide 60 302 286 302 $script:Gold 5 | Out-Null
    Add-Text $slide "Murtada Asim Ibrahim Bakhit (18-317)" 60 345 430 24 17 $script:OffWhite | Out-Null
    Add-Text $slide "Suliman Elawad Suliman Babiker (18-310)" 60 374 430 24 17 $script:OffWhite | Out-Null
    Add-Text $slide "Abdalhay Abass Mobarak Abass (15-306)" 60 403 430 24 17 $script:OffWhite | Out-Null
    Add-Text $slide "Offline research simulation  |  August 2026" 60 470 480 24 16 $script:Muted | Out-Null
    Add-Notes $slide (New-Notes "Can PPO allocate limited modeled supply more effectively than Fixed Rotation?" @(
            "docs/source/graduation-project-proposal.pdf",
            "docs/REPORT.md"
        ))

    # 2 - problem
    $slide = $presentation.Slides.Add(2, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 2
    Add-SlideTitle $slide "Seven simultaneous allocation choices under shortage"
    Add-BulletList $slide @(
        "Protect Hospital, Water, and Bakery demand",
        "Use available supply without exceeding it",
        "Share service across seven modeled localities"
    ) 58 160 330 21 $script:OffWhite 66 | Out-Null
    Add-Box $slide 425 140 485 273 $script:Slate $script:Grid 5 | Out-Null
    Add-Picture $slide $overviewImage 432 147 471 265 "Overview showing seven administrative locality boundaries and frozen outcomes" | Out-Null
    Add-Text $slide "Synthetic decision support - not a live grid controller" 58 438 790 30 17 $script:Teal $true | Out-Null
    Add-Notes $slide (New-Notes "Administrative locality boundaries provide presentation context and are not electrical-network topology." @(
            "docs/METHODOLOGY.md",
            "docs/assets/screenshots/overview.png"
        ))

    # 3 - system boundary
    $slide = $presentation.Slides.Add(3, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 3 $true
    Add-SlideTitle $slide "The controller evidence path is isolated from presentation context" $true
    $nodeCenters = @(125, 300, 475, 650, 825)
    for ($index = 0; $index -lt 4; $index++) {
        Add-Line $slide ($nodeCenters[$index] + 62) 230 ($nodeCenters[$index + 1] - 62) 230 $script:Gold 2.5 $true | Out-Null
    }
    $nodeLabels = @(
        ("Seeded" + [Environment]::NewLine + "scenarios"),
        ("Shared CUDA" + [Environment]::NewLine + "tensors"),
        ("PPO and" + [Environment]::NewLine + "Fixed Rotation"),
        ("Paired" + [Environment]::NewLine + "evidence"),
        ("Desktop" + [Environment]::NewLine + "replay")
    )
    for ($index = 0; $index -lt $nodeCenters.Count; $index++) {
        $fill = if ($index -in @(2, 4)) { $script:Teal } else { $script:Slate }
        Add-Box $slide ($nodeCenters[$index] - 62) 184 124 92 $fill "" 5 | Out-Null
        Add-Text $slide $nodeLabels[$index] ($nodeCenters[$index] - 55) 203 110 64 17 $script:OffWhite $true 2 | Out-Null
    }
    Add-Text $slide "Controller evidence" 58 320 230 28 20 $script:Ink $true | Out-Null
    Add-Line $slide 58 355 900 355 $script:Grid 1 | Out-Null
    Add-Text $slide "Demand projection" 58 380 190 28 18 $script:Ink $true | Out-Null
    Add-Text $slide "Display only - never enters PPO, dispatch, training objective, or benchmark evidence" 250 380 660 50 18 "60727D" | Out-Null
    Add-Text $slide "Offline after installation  |  hash-bound artifacts  |  fail-closed startup" 58 448 820 24 18 $script:Teal $true | Out-Null
    Add-Notes $slide (New-Notes "The controller path and display-only forecast remain separate so presentation context cannot silently change the evidence." @(
            "docs/IMPLEMENTATION.md",
            "docs/decisions/ADR-001-gpu-runtime.md"
        ))

    # 4 - matched method
    $slide = $presentation.Slides.Add(4, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 4
    Add-SlideTitle $slide "PPO and Fixed Rotation receive the same scenario"
    Add-StatCard $slide 62 156 236 126 "$baseDrawCount independent base draws" "10 untouched seeds x 32 episodes" $script:Slate
    Add-Text $slide "x 3" 310 192 80 42 30 $script:Gold $true 2 | Out-Null
    Add-StatCard $slide 400 156 218 126 "$matchedScenarioCount matched" "mild, moderate, severe shortage" $script:Teal $script:Navy
    Add-Text $slide "x 2" 630 192 80 42 30 $script:Gold $true 2 | Out-Null
    Add-StatCard $slide 720 156 178 126 "$controllerEvaluationCount" "controller-scenario evaluations" $script:Slate
    Add-Line $slide 62 318 898 318 $script:Grid 1 | Out-Null
    Add-Text $slide "PPO" 70 350 175 30 24 $script:Teal $true | Out-Null
    Add-Text $slide ("32-value observation" + [Environment]::NewLine + "7-locality joint action") 70 386 290 54 18 $script:OffWhite | Out-Null
    Add-Text $slide "Fixed Rotation" 515 350 230 30 24 $script:Gold $true | Out-Null
    Add-Text $slide ("Three-hour schedule blocks" + [Environment]::NewLine + "No demand or fairness observation") 515 386 330 54 18 $script:OffWhite | Out-Null
    Add-Text $slide "Every stored delta is PPO - Fixed Rotation" 250 458 460 26 19 $script:Teal $true 2 | Out-Null
    Add-Notes $slide (New-Notes "The 960 conditions are severity variants of 320 independent base draws. Both controllers receive the same immutable scenario." @(
            "docs/METHODOLOGY.md",
            "data/evidence/robustness-supplement.json, SHA-256 $expectedRobustnessSha256"
        ))

    # 5 - training record
    $slide = $presentation.Slides.Add(5, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 5 $true
    Add-SlideTitle $slide "The PPO checkpoint has a historical training record" $true
    Add-Box $slide 58 148 844 78 "E4F4F0" $script:Teal 5 | Out-Null
    Add-Text $slide "Network" 76 163 110 22 16 "60727D" $true | Out-Null
    Add-Text $slide ([string]$trainingSummary.network) 76 188 250 28 21 $script:Ink $true | Out-Null
    Add-Text $slide "Selected checkpoint" 354 163 190 22 16 "60727D" $true | Out-Null
    Add-Text $slide ("iteration {0}  |  seed {1}" -f $trainingSummary.selection.selected_iteration, $trainingSummary.selection.selected_seed) 354 188 260 28 21 $script:Ink $true | Out-Null
    Add-Text $slide "Validation seeds" 658 163 150 22 16 "60727D" $true | Out-Null
    Add-Text $slide (($trainingSummary.selection.validation_seeds -join ", ")) 658 188 220 28 20 $script:Ink $true | Out-Null
    Add-Text $slide "Historical initialization" 64 258 300 28 21 $script:Teal $true | Out-Null
    Add-BulletList $slide @(
        "$($trainingSummary.behavior_cloning_initialization.epochs) epochs  |  $($trainingSummary.behavior_cloning_initialization.teacher_samples) teacher samples",
        "$($trainingSummary.behavior_cloning_initialization.optimizer_updates) initialization optimizer updates",
        "$($trainingSummary.ppo.iterations) PPO iterations  |  $($trainingSummary.ppo.total_episodes) episodes",
        "$($trainingSummary.ppo.total_transitions) transitions  |  learning rate $($trainingSummary.ppo.learning_rate)"
    ) 64 298 420 16 $script:Ink 40 $script:Teal | Out-Null
    Add-Text $slide "Training-objective score (diagnostic)" 536 258 350 28 20 $script:Gold $true | Out-Null
    Add-Text $slide (Format-TwoDecimals ([double]$trainingSummary.convergence.selected_iteration_510)) 536 302 170 58 38 $script:Teal $true | Out-Null
    Add-Text $slide "selected iteration 510" 536 358 210 24 16 "60727D" | Out-Null
    Add-Text $slide (Format-TwoDecimals ([double]$trainingSummary.convergence.untrained)) 748 302 140 58 38 "60727D" $true | Out-Null
    Add-Text $slide "untrained" 748 358 140 24 16 "60727D" | Out-Null
    Add-Text $slide "Model selection used validation seeds; untouched evaluation seeds were not used for training or selection." 536 404 350 68 17 $script:Ink $true | Out-Null
    Add-Notes $slide (New-Notes "The historical teacher-guided initialization precedes PPO training; the deployed controller remains PPO." @(
            "data/model/training-provenance.yaml, SHA-256 $expectedTrainingSha256",
            "data/model/manifest.json"
        ))

    # 6 - GPU execution
    $slide = $presentation.Slides.Add(6, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 6
    Add-SlideTitle $slide "GPU-first simulation, PPO inference, and paired bootstrap"
    Add-Line $slide 200 237 365 237 $script:Gold 2.5 $true | Out-Null
    Add-Line $slide 535 237 700 237 $script:Gold 2.5 $true | Out-Null
    Add-Box $slide 70 172 190 128 $script:Slate "" 5 | Out-Null
    Add-Text $slide "PyTorch" 85 193 160 32 25 $script:OffWhite $true 2 | Out-Null
    Add-Text $slide ("scenario tensors" + [Environment]::NewLine + "simulation" + [Environment]::NewLine + "PPO inference") 85 231 160 64 16 $script:Muted $false 2 | Out-Null
    Add-Box $slide 385 155 166 164 $script:Teal "" 5 | Out-Null
    Add-Text $slide "cuda:0" 402 198 132 44 31 $script:Navy $true 2 | Out-Null
    Add-Text $slide ("validated" + [Environment]::NewLine + "NVIDIA GPU") 401 251 134 48 16 $script:Navy $true 2 | Out-Null
    Add-Box $slide 690 172 200 128 $script:Slate "" 5 | Out-Null
    Add-Text $slide "CuPy" 708 193 164 32 25 $script:OffWhite $true 2 | Out-Null
    Add-Text $slide ("cluster bootstrap" + [Environment]::NewLine + "CUDA sort" + [Environment]::NewLine + "linear percentiles") 708 231 164 64 16 $script:Muted $false 2 | Out-Null
    Add-BulletList $slide @(
        "PyTorch 2.8 + CUDA 12.8  |  DLPack sharing with CuPy",
        "10,000 deterministic seed-stratified bootstrap replicates",
        "One final host transfer assembles evidence; it is not a controller fallback"
    ) 70 365 820 17 $script:OffWhite 37 $script:Teal | Out-Null
    Add-Notes $slide (New-Notes "Simulation, PPO inference, metrics, and bootstrap stay GPU-first. One final host transfer supports portable artifact assembly." @(
            "docs/IMPLEMENTATION.md",
            "docs/decisions/ADR-001-gpu-runtime.md"
        ))

    # 7 - live desktop
    $slide = $presentation.Slides.Add(7, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 7 $true
    Add-SlideTitle $slide "The desktop replays one audited GPU scenario" $true
    Add-Box $slide 62 126 836 352 "D7E0E4" $script:Grid 5 | Out-Null
    Add-Picture $slide $liveImage 69 133 822 338 "Live Scenario page showing one shared GPU scenario and locality detail" | Out-Null
    Add-Text $slide "Frozen evidence stays separate from Live Scenario replay  |  forecast is DISPLAY ONLY" 75 478 810 24 16 $script:Teal $true 2 | Out-Null
    Add-Notes $slide (New-Notes "Run once, then change controller and hour without changing the scenario. Live values never replace frozen evidence." @(
            "docs/IMPLEMENTATION.md",
            "docs/assets/screenshots/live.png"
        ))

    # 8 - untouched outcomes
    $slide = $presentation.Slides.Add(8, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 8 $true
    Add-SlideTitle $slide "Untouched confirmatory evidence reports three primary outcomes" $true
    Add-Text $slide "Outcome" 64 145 226 24 16 "60727D" $true | Out-Null
    Add-Text $slide "Fixed Rotation" 300 145 126 24 16 "60727D" $true | Out-Null
    Add-Text $slide "PPO" 434 145 126 24 16 "60727D" $true | Out-Null
    Add-Text $slide "Delta" 568 145 112 24 16 "60727D" $true | Out-Null
    Add-Text $slide "Paired 95% interval" 688 145 204 24 16 "60727D" $true | Out-Null
    Add-OutcomeRow $slide 183 "Critical-outage locality-hours" ((Format-OneDecimal ([double]$fixed.critical_outage_locality_hours)) + " locality-hours") ((Format-OneDecimal ([double]$ppo.critical_outage_locality_hours)) + " locality-hours") $outageDelta $outageInterval $true
    Add-OutcomeRow $slide 246 "Jain fairness" (Format-Percent ([double]$fixed.fairness)) (Format-Percent ([double]$ppo.fairness)) $fairnessDelta $fairnessInterval
    Add-OutcomeRow $slide 309 "Utilization" (Format-Percent ([double]$fixed.utilization)) (Format-Percent ([double]$ppo.utilization)) $utilizationDelta $utilizationInterval $true
    Add-Line $slide 62 377 898 377 "CFD8DC" 1 | Out-Null
    Add-Text $slide "Safety check" 64 397 150 24 17 $script:Ink $true | Out-Null
    Add-Text $slide "$powerBalanceLabel power-balance violation rate in both controllers" 220 397 540 24 18 $script:Teal $true | Out-Null
    Add-Box $slide 64 443 828 38 "E4F4F0" $script:Teal 5 | Out-Null
    Add-Text $slide "Observed as-is  |  no favorable/adverse release gate" 80 452 796 22 17 $script:Ink $true 2 | Out-Null
    Add-Notes $slide (New-Notes "These are the only three headline outcomes. They use 10 untouched seeds and were reported as observed; result direction was not a release condition." @(
            "data/evidence/robustness-supplement.json, SHA-256 $expectedRobustnessSha256",
            "docs/RESULTS.md"
        ))

    # 9 - sensitivity
    $slide = $presentation.Slides.Add(9, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 9
    Add-SlideTitle $slide "24 demand-allocation variants preserve PPO's direction"
    Add-Text $slide "$sensitivityParameterizationCount parameterizations" 64 145 330 42 29 $script:Gold $true | Out-Null
    Add-Text $slide "1 locked base + 24 one-at-a-time variants" 64 190 390 28 18 $script:OffWhite | Out-Null
    Add-StatCard $slide 64 246 246 126 "$($sensitivityFavorable.critical_outage_locality_hours)/24" "critical-outage locality-hours favorable" $script:Slate
    Add-StatCard $slide 357 246 246 126 "$($sensitivityFavorable.fairness)/24" "Jain fairness favorable" $script:Slate
    Add-StatCard $slide 650 246 246 126 "$($sensitivityFavorable.utilization)/24" "utilization favorable" $script:Slate
    Add-Text $slide "Effect-size shift versus base" 64 400 260 24 17 $script:Teal $true | Out-Null
    Add-Text $slide "Outage: $outageDidAdverse adverse  |  $outageDidFavorable favorable  |  $outageDidAmbiguous ambiguous" 330 400 560 24 17 $script:OffWhite $true | Out-Null
    Add-Text $slide "Direction persists, but effect size is not invariant. Classifications are evidence, not a release gate." 64 446 826 40 17 $script:Muted | Out-Null
    Add-Notes $slide (New-Notes "All 24 variants retain favorable paired direction for each primary outcome. Difference-in-differences is mixed and no result-direction gate filtered the artifact." @(
            "data/evidence/robustness-supplement.json, sensitivity",
            "docs/RESULTS.md"
        ))

    # 10 - forecast
    $slide = $presentation.Slides.Add(10, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 10 $true
    Add-SlideTitle $slide "One-hour persistence outperformed the three-hour trailing mean" $true
    Add-Text $slide "Pooled error metric" 64 144 285 24 16 "60727D" $true | Out-Null
    Add-Text $slide "One-hour persistence" 394 144 220 24 16 $script:Teal $true | Out-Null
    Add-Text $slide "Three-hour trailing mean" 655 144 235 24 16 "60727D" $true | Out-Null
    $forecastRows = @(
        @("Locality-sector MAE", $persistence.locality_sector.mae, $trailingMean.locality_sector.mae),
        @("Locality-sector RMSE", $persistence.locality_sector.rmse, $trailingMean.locality_sector.rmse),
        @("System-total MAE", $persistence.system_total.mae, $trailingMean.system_total.mae),
        @("System-total RMSE", $persistence.system_total.rmse, $trailingMean.system_total.rmse)
    )
    for ($index = 0; $index -lt $forecastRows.Count; $index++) {
        $top = 184 + ($index * 54)
        if ($index % 2 -eq 0) {
            Add-Box $slide 54 ($top - 7) 852 44 "E7ECEE" "" 1 | Out-Null
        }
        Add-Text $slide $forecastRows[$index][0] 64 $top 285 28 18 $script:Ink $true | Out-Null
        Add-Text $slide (Format-TwoDecimals ([double]$forecastRows[$index][1])) 394 $top 220 28 19 $script:Teal $true | Out-Null
        Add-Text $slide (Format-TwoDecimals ([double]$forecastRows[$index][2])) 655 $top 220 28 19 "60727D" $true | Out-Null
    }
    Add-Box $slide 64 418 828 62 "E4F4F0" $script:Teal 5 | Out-Null
    Add-Text $slide "$($forecast.trace_count) traces  |  $(@($forecast.target_hours).Count) target hours  |  $($forecast.locality_sector_prediction_count) locality-sector predictions  |  $($forecast.system_total_prediction_count) system totals" 82 431 792 20 16 $script:Ink $true 2 | Out-Null
    Add-Text $slide "Synthetic display-only comparison; no operational calibration claim" 82 455 792 20 16 "60727D" $true 2 | Out-Null
    Add-Notes $slide (New-Notes "Lower error for one-hour persistence is a bounded synthetic display comparison, not operational calibration." @(
            "data/evidence/robustness-supplement.json, forecast_evaluation",
            "docs/METHODOLOGY.md"
        ))

    # 11 - accepted reproduction
    $slide = $presentation.Slides.Add(11, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 11
    Add-SlideTitle $slide "The accepted reproduction preserves history, not independence"
    Add-Text $slide "Accepted reproduction context" 64 145 360 30 23 $script:Teal $true | Out-Null
    Add-BulletList $slide @(
        "Seeds 50001-50003  |  96 base draws",
        "288 matched conditions  |  576 evaluations",
        "Reproduced from the frozen PPO checkpoint",
        "Not untouched: promotion and acceptance used this layer"
    ) 64 190 410 17 $script:OffWhite 47 $script:Teal | Out-Null
    Add-Line $slide 500 142 500 457 $script:Grid 1 | Out-Null
    Add-Text $slide "Accepted three-primary context" 538 145 345 30 22 $script:Gold $true | Out-Null
    Add-Text $slide "Critical outage" 538 199 175 24 16 $script:Muted $true | Out-Null
    Add-Text $slide ("{0} -> {1} locality-hours" -f (Format-OneDecimal ([double]$acceptedFixed.critical_outage_locality_hours)), (Format-OneDecimal ([double]$acceptedPpo.critical_outage_locality_hours))) 538 225 330 28 19 $script:OffWhite $true | Out-Null
    Add-Text $slide "Jain fairness" 538 273 175 24 16 $script:Muted $true | Out-Null
    Add-Text $slide ("{0} -> {1}" -f (Format-Percent ([double]$acceptedFixed.fairness)), (Format-Percent ([double]$acceptedPpo.fairness))) 538 299 330 28 19 $script:OffWhite $true | Out-Null
    Add-Text $slide "Utilization" 538 347 175 24 16 $script:Muted $true | Out-Null
    Add-Text $slide ("{0} -> {1}" -f (Format-Percent ([double]$acceptedFixed.utilization)), (Format-Percent ([double]$acceptedPpo.utilization))) 538 373 330 28 19 $script:OffWhite $true | Out-Null
    Add-Box $slide 538 418 330 55 $script:SoftSlate $script:Grid 5 | Out-Null
    Add-Text $slide "Untouched seeds 81001-81010 remain separate" 552 427 302 40 16 $script:Teal $true 2 | Out-Null
    Add-Notes $slide (New-Notes "The accepted artifact preserves provenance, but acceptance used its seeds. Confirmatory claims use the separate untouched seed layer." @(
            "data/evidence/paired-benchmark.json, SHA-256 $expectedPairedSha256",
            "data/evidence/robustness-supplement.json, seed_separation"
        ))

    # 12 - limitations
    $slide = $presentation.Slides.Add(12, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 12 $true
    Add-SlideTitle $slide "The evidence remains bounded by a shortage-only synthetic model" $true
    Add-Text $slide "What the evidence supports" 64 145 360 30 23 $script:Teal $true | Out-Null
    Add-BulletList $slide @(
        "PPO versus Fixed Rotation in the locked simulator",
        "Matched shortage scenarios across seven localities",
        "Reproducible artifacts with explicit seed roles",
        "Offline decision-support demonstration"
    ) 64 192 410 17 $script:Ink 50 $script:Teal | Out-Null
    Add-Text $slide "What it does not support" 522 145 370 30 23 $script:Risk $true | Out-Null
    Add-BulletList $slide @(
        "Field performance on Sudan's electrical grid",
        "Voltage, frequency, protection, or network stability",
        "Real-data forecast calibration",
        "Automatic operational dispatch"
    ) 522 192 390 17 $script:Ink 50 $script:Risk | Out-Null
    Add-Box $slide 64 424 828 52 "F6E8E3" $script:Risk 5 | Out-Null
    Add-Text $slide "Synthetic demand and administrative geography are abstractions, not geographic measurements or electrical topology." 82 431 790 40 16 $script:Ink $true 2 | Out-Null
    Add-Notes $slide (New-Notes "The experiment models shortage allocation, not a power-flow network, and the data are synthetic." @(
            "docs/METHODOLOGY.md",
            "docs/RESULTS.md"
        ))

    # 13 - conclusion
    $slide = $presentation.Slides.Add(13, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 13
    Add-Text $slide "THE ANSWER" 60 72 240 24 16 $script:Teal $true | Out-Null
    Add-Text $slide "PPO outperformed Fixed Rotation" 60 128 820 70 46 $script:OffWhite $true | Out-Null
    Add-Text $slide "under the locked synthetic evaluation protocol" 62 205 700 34 24 $script:Gold | Out-Null
    Add-Line $slide 62 271 898 271 $script:Grid 1 | Out-Null
    Add-BulletList $slide @(
        "Three untouched primary outcomes - reported without a favorable-result gate",
        "Matched scenarios, explicit seed roles, and hash-bound evidence",
        "GPU-first offline desktop with Frozen and Live views kept separate"
    ) 62 306 820 18 $script:OffWhite 49 | Out-Null
    Add-Box $slide 62 452 836 42 $script:Teal "" 5 | Out-Null
    Add-Text $slide "Next: representative operational data and electrical-network constraints" 78 463 804 22 17 $script:Navy $true 2 | Out-Null
    Add-Notes $slide (New-Notes "PPO outperformed Fixed Rotation under this locked synthetic protocol. Operational-data and network validation must precede field use." @(
            "docs/RESULTS.md",
            "docs/REPORT.md"
        ))

    Assert-PresentationLayout $presentation
    $presentation.SaveAs($OutputPath, 24)
    Get-ChildItem -LiteralPath $renderDirectory -Filter "slide-*.png" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force
    for ($index = 1; $index -le $presentation.Slides.Count; $index++) {
        $renderPath = Join-Path $renderDirectory ("slide-{0:D2}.png" -f $index)
        $presentation.Slides.Item($index).Export($renderPath, "PNG", 1600, 900)
    }
    Write-Output "Presentation written: $OutputPath"
    Write-Output "Rendered slides: $renderDirectory"
}
finally {
    if ($null -ne $presentation) {
        $presentation.Close()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    }
    if ($null -ne $powerPoint) {
        $powerPoint.Quit()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
