param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

# Native PowerPoint automation preserves editable text, shapes, speaker notes, and slide
# structure while also exporting deterministic review images for every slide.

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $projectRoot "output\presentation\gridguard-khartoum-defense.pptx"
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$renderDirectory = Join-Path $projectRoot "tmp\presentation\rendered"
$overviewImage = Join-Path $projectRoot "docs\assets\screenshots\overview.png"
$liveImage = Join-Path $projectRoot "docs\assets\screenshots\live.png"
$pairedEvidencePath = Join-Path $projectRoot "data\evidence\paired-benchmark.json"
$expectedEvidenceSha256 = "2c3e71e48c4c7bff4fe609d42a76ca4f770f6dd717cd7d45d5e8e0d99fdf2d11"

foreach ($requiredPath in @($overviewImage, $liveImage, $pairedEvidencePath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Missing presentation image: $requiredPath"
    }
}

$actualEvidenceSha256 = (Get-FileHash -LiteralPath $pairedEvidencePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualEvidenceSha256 -cne $expectedEvidenceSha256) {
    throw "Paired evidence SHA-256 mismatch: expected $expectedEvidenceSha256, got $actualEvidenceSha256"
}
$evidence = [System.IO.File]::ReadAllText($pairedEvidencePath) | ConvertFrom-Json
if ($evidence.schema_version -ne 1) {
    throw "Unsupported paired evidence schema version: $($evidence.schema_version)"
}
if ((@($evidence.protocol.controllers) | Sort-Object) -join "|" -cne "Fixed Rotation|PPO") {
    throw "Paired evidence must contain exactly PPO and Fixed Rotation"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null
New-Item -ItemType Directory -Force -Path $renderDirectory | Out-Null

$script:SlideWidth = 960.0
$script:SlideHeight = 540.0
$script:Navy = "071923"
$script:Slate = "102A3B"
$script:Teal = "3FC6AC"
$script:Gold = "E0B957"
$script:OffWhite = "F3F0E8"
$script:Ink = "13232E"
$script:Muted = "83939D"
$script:SoftSlate = "1A394A"
$script:Grid = "365568"

function ConvertTo-Rgb {
    param([Parameter(Mandatory = $true)][string]$Hex)

    $clean = $Hex.TrimStart("#")
    if ($clean.Length -ne 6) {
        throw "Expected a six-digit hex color, got: $Hex"
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
        $Slide,
        [string]$Text,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [double]$FontSize,
        [string]$Color,
        [bool]$Bold = $false,
        [int]$Alignment = 1,
        [string]$FontName = "Aptos"
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
    # PowerPoint may collapse a newly populated text box to its one-line default height.
    # Restore the requested geometry after assigning the text and font properties.
    $shape.Width = $Width
    $shape.Height = $Height
    return $shape
}

function Add-Box {
    param(
        $Slide,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [string]$FillColor,
        [string]$LineColor = "",
        [int]$Geometry = 5,
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
        $Slide,
        [double]$X1,
        [double]$Y1,
        [double]$X2,
        [double]$Y2,
        [string]$Color,
        [double]$Weight = 1.5,
        [bool]$Arrow = $false
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
        $Slide,
        [string]$Path,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [string]$AltText
    )

    $picture = $Slide.Shapes.AddPicture($Path, 0, -1, $Left, $Top, $Width, $Height)
    $picture.AlternativeText = $AltText
    return $picture
}

function Add-Chrome {
    param($Slide, [int]$Number, [bool]$Light = $false)

    $foreground = if ($Light) { $script:Ink } else { $script:OffWhite }
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
        $Slide,
        [string[]]$Items,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$FontSize = 20,
        [string]$Color = "F3F0E8",
        [double]$Gap = 50,
        [string]$Accent = "E0B957"
    )

    for ($index = 0; $index -lt $Items.Count; $index++) {
        $itemTop = $Top + ($index * $Gap)
        Add-Line $Slide $Left ($itemTop + 12) ($Left + 16) ($itemTop + 12) $Accent 3 | Out-Null
        Add-Text $Slide $Items[$index] ($Left + 28) $itemTop ($Width - 28) ($Gap - 4) $FontSize $Color | Out-Null
    }
}

function Add-Notes {
    param($Slide, [string]$Notes)

    $notesPage = $Slide.NotesPage
    $body = $null
    for ($index = 1; $index -le $notesPage.Shapes.Count; $index++) {
        $candidate = $notesPage.Shapes.Item($index)
        if ($candidate.HasTextFrame -eq -1 -and $candidate.TextFrame.HasText -eq -1) {
            if ($candidate.Name -match "Text Placeholder|Body") {
                $body = $candidate
                break
            }
        }
    }
    if ($null -eq $body) {
        $body = $notesPage.Shapes.Placeholders.Item(2)
    }
    $body.TextFrame.TextRange.Text = $Notes
}

function Add-ComparisonBar {
    param(
        $Slide,
        [string]$Label,
        [double]$Top,
        [double]$FixedFraction,
        [double]$PpoFraction,
        [string]$FixedLabel,
        [string]$PpoLabel,
        [bool]$Light = $false
    )

    $textColor = if ($Light) { $script:Ink } else { $script:OffWhite }
    $trackColor = if ($Light) { "DDE5E8" } else { $script:SoftSlate }
    Add-Text $Slide $Label 70 $Top 190 26 18 $textColor $true | Out-Null
    Add-Box $Slide 270 ($Top + 2) 500 12 $trackColor "" 1 | Out-Null
    Add-Box $Slide 270 ($Top + 2) (500 * $FixedFraction) 12 "71838D" "" 1 | Out-Null
    Add-Box $Slide 270 ($Top + 23) 500 12 $trackColor "" 1 | Out-Null
    Add-Box $Slide 270 ($Top + 23) (500 * $PpoFraction) 12 $script:Teal "" 1 | Out-Null
    Add-Text $Slide $FixedLabel 780 ($Top - 4) 120 22 16 $(if ($Light) { "60727D" } else { $script:Muted }) | Out-Null
    Add-Text $Slide $PpoLabel 780 ($Top + 17) 120 22 16 $textColor $true | Out-Null
}

function Format-Percent {
    param([double]$Value)

    return (($Value * 100).ToString("0.00", [Globalization.CultureInfo]::InvariantCulture) + "%")
}

function Format-Points {
    param([double]$Value, [bool]$Signed = $true)

    $scaled = $Value * 100
    $prefix = if ($Signed -and $scaled -gt 0) { "+" } else { "" }
    return ($prefix + $scaled.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture) + " pp")
}

function Format-Energy {
    param([double]$Value, [bool]$Signed = $false)

    $prefix = if ($Signed -and $Value -gt 0) { "+" } else { "" }
    return ($prefix + $Value.ToString("N0", [Globalization.CultureInfo]::InvariantCulture) + " MWh")
}

function Format-Decimal {
    param([double]$Value, [bool]$Signed = $false)

    $prefix = if ($Signed -and $Value -gt 0) { "+" } else { "" }
    return ($prefix + $Value.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture))
}

function Format-DecimalInterval {
    param([double]$Lower, [double]$Upper)

    return "[{0}, {1}]" -f (Format-Decimal $Lower), (Format-Decimal $Upper)
}

function Format-PointsInterval {
    param([double]$Lower, [double]$Upper)

    $culture = [Globalization.CultureInfo]::InvariantCulture
    return "[{0}, {1}] pp" -f (($Lower * 100).ToString("0.00", $culture)), (($Upper * 100).ToString("0.00", $culture))
}

function Format-EnergyInterval {
    param([double]$Lower, [double]$Upper)

    $culture = [Globalization.CultureInfo]::InvariantCulture
    return "[{0}, {1}] MWh" -f ($Lower.ToString("N0", $culture)), ($Upper.ToString("N0", $culture))
}

function Count-FavorableCells {
    param([string]$MetricKey)

    $definition = $evidence.metric_definitions.$MetricKey
    $count = 0
    foreach ($cell in @($evidence.cells)) {
        $delta = [double]$cell.paired_delta.$MetricKey
        if (
            ($definition.direction -ceq "higher_is_better" -and $delta -gt 0) -or
            ($definition.direction -ceq "lower_is_better" -and $delta -lt 0)
        ) {
            $count += 1
        }
    }
    return $count
}

$fixed = $evidence.aggregate.'Fixed Rotation'
$ppo = $evidence.aggregate.PPO
$inference = $evidence.paired_inference
$independentBaseDrawCount = [int]$evidence.protocol.independent_base_draw_count
$shortageVariantCount = [int]$evidence.protocol.severity_variants_per_base_draw
$matchedScenarioCount = [int]$evidence.protocol.matched_scenario_count
$controllerCount = @($evidence.protocol.controllers).Count
$controllerEvaluationCount = [int]$evidence.protocol.controller_evaluation_count
$cellCount = [int]$evidence.protocol.cell_count
$bootstrapReplicates = [int]$evidence.protocol.bootstrap.replicates

if ($independentBaseDrawCount * $shortageVariantCount -ne $matchedScenarioCount) {
    throw "Paired evidence count identity is invalid"
}
if ($matchedScenarioCount * $controllerCount -ne $controllerEvaluationCount) {
    throw "Paired evidence controller-evaluation count is invalid"
}

$criticalFixedLabel = Format-Percent ([double]$fixed.critical_service_ratio)
$criticalPpoLabel = Format-Percent ([double]$ppo.critical_service_ratio)
$criticalDeltaLabel = Format-Points ([double]$inference.critical_service_ratio.mean_delta)
$fixedOutageLabel = Format-Decimal ([double]$fixed.critical_outage_locality_hours)
$ppoOutageLabel = Format-Decimal ([double]$ppo.critical_outage_locality_hours)

$fixedServedLabel = Format-Energy ([double]$fixed.served_mwh)
$ppoServedLabel = Format-Energy ([double]$ppo.served_mwh)
$servedDeltaLabel = Format-Energy ([double]$inference.served_mwh.mean_delta) $true
$servedDeltaMagnitudeLabel = Format-Energy ([Math]::Abs([double]$inference.served_mwh.mean_delta))
$fixedUtilizationLabel = Format-Percent ([double]$fixed.utilization)
$ppoUtilizationLabel = Format-Percent ([double]$ppo.utilization)
$utilizationDeltaLabel = Format-Points ([double]$inference.utilization.mean_delta)
$fixedCurtailedLabel = Format-Energy ([double]$fixed.curtailed_mwh)
$ppoCurtailedLabel = Format-Energy ([double]$ppo.curtailed_mwh)
$curtailedDeltaLabel = Format-Energy ([double]$inference.curtailed_mwh.mean_delta) $true
$switchingDeltaLabel = Format-Points ([double]$inference.switching_rate.mean_delta)

$primaryMetricKeys = @(
    "episode_reward",
    "critical_service_ratio",
    "utilization",
    "fairness",
    "served_mwh",
    "curtailed_mwh"
)
$primaryMetricLabels = @(
    "Reward",
    "Critical service",
    "Utilization",
    "Fairness",
    "Served energy",
    "Curtailed energy"
)
$primaryFavorableCounts = @()
foreach ($metricKey in $primaryMetricKeys) {
    $primaryFavorableCounts += Count-FavorableCells $metricKey
}
$switchingFavorableCount = Count-FavorableCells "switching_rate"
$allPrimaryCellsFavorable = ($primaryFavorableCounts | Where-Object { $_ -ne $cellCount }).Count -eq 0
if (-not $allPrimaryCellsFavorable) {
    throw "Paired evidence no longer supports the all-primary-cells-favorable slide claim"
}

$powerBalanceRate = [Math]::Max(
    [double]$fixed.power_balance_violation_rate,
    [double]$ppo.power_balance_violation_rate
)
$powerBalanceLabel = Format-Percent $powerBalanceRate

$rewardInterval = Format-DecimalInterval `
    -Lower ([double]$inference.episode_reward.ci95_lower) `
    -Upper ([double]$inference.episode_reward.ci95_upper)
$criticalInterval = Format-PointsInterval `
    -Lower ([double]$inference.critical_service_ratio.ci95_lower) `
    -Upper ([double]$inference.critical_service_ratio.ci95_upper)
$utilizationInterval = Format-PointsInterval `
    -Lower ([double]$inference.utilization.ci95_lower) `
    -Upper ([double]$inference.utilization.ci95_upper)
$fairnessInterval = Format-PointsInterval `
    -Lower ([double]$inference.fairness.ci95_lower) `
    -Upper ([double]$inference.fairness.ci95_upper)
$servedInterval = Format-EnergyInterval `
    -Lower ([double]$inference.served_mwh.ci95_lower) `
    -Upper ([double]$inference.served_mwh.ci95_upper)
$curtailedInterval = Format-EnergyInterval `
    -Lower ([double]$inference.curtailed_mwh.ci95_lower) `
    -Upper ([double]$inference.curtailed_mwh.ci95_upper)
$switchingInterval = Format-PointsInterval `
    -Lower ([double]$inference.switching_rate.ci95_lower) `
    -Upper ([double]$inference.switching_rate.ci95_upper)

$intervalRows = @(
    @(
        "Episode reward",
        (Format-Decimal ([double]$inference.episode_reward.mean_delta) $true),
        $rewardInterval
    ),
    @(
        "Critical service",
        (Format-Points ([double]$inference.critical_service_ratio.mean_delta)),
        $criticalInterval
    ),
    @(
        "Utilization",
        (Format-Points ([double]$inference.utilization.mean_delta)),
        $utilizationInterval
    ),
    @(
        "Jain fairness",
        (Format-Points ([double]$inference.fairness.mean_delta)),
        $fairnessInterval
    ),
    @(
        "Served energy",
        (Format-Energy ([double]$inference.served_mwh.mean_delta) $true),
        $servedInterval
    ),
    @(
        "Curtailed energy",
        (Format-Energy ([double]$inference.curtailed_mwh.mean_delta) $true),
        $curtailedInterval
    ),
    @(
        "Switching rate",
        (Format-Points ([double]$inference.switching_rate.mean_delta)),
        $switchingInterval
    )
)

$powerPoint = $null
$presentation = $null

try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $presentation = $powerPoint.Presentations.Add()
    $presentation.PageSetup.SlideWidth = $script:SlideWidth
    $presentation.PageSetup.SlideHeight = $script:SlideHeight

    # Slide 1 - title
    $slide = $presentation.Slides.Add(1, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Text $slide "GRADUATION PROJECT DEFENSE" 60 58 420 24 16 $script:Teal $true | Out-Null
    Add-Text $slide "GridGuard Khartoum" 60 128 780 70 54 $script:OffWhite $true | Out-Null
    Add-Text $slide "PPO-assisted load shedding and energy management for developing countries" 60 212 760 68 26 $script:Gold | Out-Null
    Add-Line $slide 60 302 286 302 $script:Gold 5 | Out-Null
    Add-Text $slide "Murtada Asim Ibrahim Bakhit (18-317)" 60 345 420 24 17 $script:OffWhite | Out-Null
    Add-Text $slide "Suliman Elawad Suliman Babiker (18-310)" 60 374 420 24 17 $script:OffWhite | Out-Null
    Add-Text $slide "Abdalhay Abass Mobarak Abass (15-306)" 60 403 420 24 17 $script:OffWhite | Out-Null
    Add-Text $slide "Offline research simulation  |  August 2026" 60 470 480 24 16 $script:Muted | Out-Null
    Add-Notes $slide ("Open with the project question: can a learned policy allocate limited modeled supply more effectively than a traditional schedule?`r`n`r`n[Sources]`r`n- docs/source/graduation-project-proposal.pdf`r`n- docs/REPORT.md`r`n[/Sources]")

    # Slide 2 - problem
    $slide = $presentation.Slides.Add(2, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 2
    Add-SlideTitle $slide "A shortage creates seven simultaneous allocation choices"
    Add-BulletList $slide @(
        "Protect Hospital, Water, and Bakery demand",
        "Use the available supply without exceeding it",
        "Share service across seven modeled localities"
    ) 58 160 330 21 $script:OffWhite 66 | Out-Null
    Add-Box $slide 425 140 485 273 $script:Slate $script:Grid 5 | Out-Null
    Add-Picture $slide $overviewImage 432 147 471 265 "GridGuard overview showing seven administrative locality boundaries and the frozen controller summary" | Out-Null
    Add-Text $slide "Synthetic decision-support scope - not a live grid controller" 58 438 790 30 17 $script:Teal $true | Out-Null
    Add-Notes $slide ("Frame the problem as constrained allocation, not a claim that the simulator models the full electrical network. The administrative boundaries provide presentation context only.`r`n`r`n[Sources]`r`n- docs/METHODOLOGY.md (Simulation domain; Limitations)`r`n- docs/assets/screenshots/overview.png`r`n[/Sources]")

    # Slide 3 - system
    $slide = $presentation.Slides.Add(3, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 3 $true
    Add-SlideTitle $slide "The proposal became a clean, auditable desktop system" $true
    $nodeCenters = @(130, 305, 480, 655, 830)
    for ($index = 0; $index -lt 4; $index++) {
        Add-Line $slide ($nodeCenters[$index] + 65) 230 ($nodeCenters[$index + 1] - 65) 230 $script:Gold 2.5 $true | Out-Null
    }
    $nodeLabels = @("Seeded`nscenarios", "Shared CUDA`ntensors", "PPO and`nFixed Rotation", "Paired`nevidence", "Desktop`nreplay")
    for ($index = 0; $index -lt $nodeCenters.Count; $index++) {
        $fill = if ($index -in @(2, 4)) { $script:Teal } else { $script:Slate }
        Add-Box $slide ($nodeCenters[$index] - 65) 184 130 92 $fill "" 5 | Out-Null
        Add-Text $slide $nodeLabels[$index] ($nodeCenters[$index] - 57) 203 114 66 18 $script:OffWhite $true 2 | Out-Null
    }
    Add-Text $slide "Controller evidence path" 58 320 280 26 20 $script:Ink $true | Out-Null
    Add-Line $slide 58 355 900 355 $script:Grid 1 | Out-Null
    Add-Text $slide "Demand projection" 58 380 190 28 18 $script:Ink $true | Out-Null
    Add-Text $slide "Display only - never enters PPO, dispatch, reward, or benchmark evidence" 250 380 650 34 18 "60727D" | Out-Null
    Add-Text $slide "Offline after installation  |  hash-bound artifacts  |  fail-closed startup" 58 446 820 24 18 $script:Teal $true | Out-Null
    Add-Notes $slide ("Walk left to right. The controller path and display-only forecast are deliberately separated so a presentation feature cannot silently change the claim.`r`n`r`n[Sources]`r`n- docs/IMPLEMENTATION.md (System overview; Display-only forecast)`r`n- docs/decisions/ADR-001-gpu-runtime.md`r`n[/Sources]")

    # Slide 4 - matched design
    $slide = $presentation.Slides.Add(4, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 4
    Add-SlideTitle $slide "PPO and Fixed Rotation receive exactly the same scenario"
    Add-Text $slide ([string]$independentBaseDrawCount) 68 150 150 62 48 $script:Gold $true 2 | Out-Null
    Add-Text $slide "independent`nbase draws" 68 214 150 46 18 $script:OffWhite $true 2 | Out-Null
    Add-Text $slide "x $shortageVariantCount" 232 169 100 40 31 $script:Muted $true 2 | Out-Null
    Add-Text $slide ([string]$matchedScenarioCount) 344 150 160 62 48 $script:Teal $true 2 | Out-Null
    Add-Text $slide "matched scenario`nconditions" 344 214 160 46 18 $script:OffWhite $true 2 | Out-Null
    Add-Text $slide "x $controllerCount" 518 169 100 40 31 $script:Muted $true 2 | Out-Null
    Add-Text $slide ([string]$controllerEvaluationCount) 630 150 170 62 48 $script:Gold $true 2 | Out-Null
    Add-Text $slide "controller-scenario`nevaluations" 630 214 190 46 18 $script:OffWhite $true 2 | Out-Null
    Add-Line $slide 70 305 890 305 $script:Grid 1 | Out-Null
    Add-Text $slide "PPO" 78 340 170 30 24 $script:Teal $true | Out-Null
    Add-Text $slide "32-value causal observation`nLearned 7-locality joint action" 78 378 280 55 18 $script:OffWhite | Out-Null
    Add-Text $slide "Fixed Rotation" 508 340 230 30 24 $script:Gold $true | Out-Null
    Add-Text $slide "Three-hour schedule blocks`nNo demand or fairness observation" 508 378 330 55 18 $script:OffWhite | Out-Null
    Add-Text $slide "Every stored delta is PPO - Fixed Rotation" 250 458 460 26 19 $script:Teal $true 2 | Out-Null
    Add-Notes $slide ("Stress the pairing: both controllers receive the same immutable Scenario object. The $matchedScenarioCount conditions are severity variants of $independentBaseDrawCount independent base draws, not $matchedScenarioCount independent draws; together they produce $controllerEvaluationCount controller-scenario evaluations.`r`n`r`n[Sources]`r`n- docs/METHODOLOGY.md (Controller decisions; Matched evaluation design)`r`n- data/evidence/paired-benchmark.json, SHA-256 $expectedEvidenceSha256`r`n[/Sources]")

    # Slide 5 - GPU
    $slide = $presentation.Slides.Add(5, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 5 $true
    Add-SlideTitle $slide "GPU-first simulation, PPO inference, and paired bootstrap" $true
    Add-Line $slide 180 236 370 236 $script:Gold 2.5 $true | Out-Null
    Add-Line $slide 505 236 695 236 $script:Gold 2.5 $true | Out-Null
    Add-Box $slide 70 178 180 116 $script:Slate "" 5 | Out-Null
    Add-Text $slide "PyTorch" 86 198 148 30 24 $script:OffWhite $true 2 | Out-Null
    Add-Text $slide "scenarios`nsimulation`nPPO inference" 86 232 148 60 16 $script:Muted $false 2 | Out-Null
    Add-Box $slide 390 157 140 158 $script:Teal "" 5 | Out-Null
    Add-Text $slide "cuda:0" 402 199 116 44 31 $script:Navy $true 2 | Out-Null
    Add-Text $slide "no CPU fallback" 398 250 124 24 16 $script:Navy $true 2 | Out-Null
    Add-Box $slide 715 178 180 116 $script:Slate "" 5 | Out-Null
    Add-Text $slide "CuPy" 731 198 148 30 24 $script:OffWhite $true 2 | Out-Null
    Add-Text $slide "cluster bootstrap`nCUDA sort`nlinear percentiles" 731 232 148 60 16 $script:Muted $false 2 | Out-Null
    Add-Text $slide "Validated presentation runtime" 70 350 330 28 20 $script:Ink $true | Out-Null
    Add-BulletList $slide @(
        "RTX 5060 Laptop GPU  |  PyTorch 2.8.0+cu128",
        "CUDA 12.8  |  CuPy 14.1.1  |  DLPack coherence",
        "Startup blocks on GPU, model, evidence, or provenance failure"
    ) 70 390 820 17 $script:Ink 33 $script:Teal | Out-Null
    Add-Notes $slide ("The performance-critical scenario, simulation, PPO, metric, and bootstrap paths run on CUDA. One explicit final host transfer supports canonical evidence assembly and validation; it is not a controller fallback. The release gate validates Torch, CuPy, CUDA, DLPack sharing, the PPO artifact, and evidence before launch.`r`n`r`n[Sources]`r`n- docs/IMPLEMENTATION.md (GPU-first execution; Verification strategy)`r`n- docs/decisions/ADR-001-gpu-runtime.md`r`n[/Sources]")

    # Slide 6 - live UI
    $slide = $presentation.Slides.Add(6, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 6
    Add-SlideTitle $slide "The desktop replays one audited GPU scenario"
    Add-Box $slide 63 123 834 355 $script:Slate $script:Grid 5 | Out-Null
    Add-Picture $slide $liveImage 70 130 820 346 "Live Scenario page showing a shared GPU run, interactive administrative map, and locality detail" | Out-Null
    Add-Text $slide "Switch controller  |  scrub 24 hours  |  inspect locality  |  forecast is DISPLAY ONLY" 78 480 804 24 16 $script:Teal $true 2 | Out-Null
    Add-Notes $slide ("Demo this after the slide: choose a shortage and seed, run once, then change controller and hour without changing the scenario. Point out demand, served MW, critical ratio, sector service, and the display-only next-hour projection.`r`n`r`n[Sources]`r`n- docs/IMPLEMENTATION.md (Desktop presentation)`r`n- docs/assets/screenshots/live.png`r`n[/Sources]")

    # Slide 7 - critical result
    $slide = $presentation.Slides.Add(7, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 7 $true
    Add-SlideTitle $slide "PPO protects nearly all modeled critical demand" $true
    Add-Text $slide $criticalPpoLabel 68 145 330 82 54 $script:Teal $true | Out-Null
    Add-Text $slide "PPO critical-service ratio" 70 226 310 28 19 $script:Ink $true | Out-Null
    Add-Text $slide $criticalFixedLabel 540 145 330 82 54 "60727D" $true | Out-Null
    Add-Text $slide "Fixed Rotation" 542 226 310 28 19 $script:Ink $true | Out-Null
    Add-ComparisonBar $slide "Critical service" 286 ([double]$fixed.critical_service_ratio) ([double]$ppo.critical_service_ratio) $criticalFixedLabel $criticalPpoLabel $true
    Add-Line $slide 70 378 890 378 "CFD8DC" 1 | Out-Null
    Add-Text $slide $criticalDeltaLabel 70 405 430 42 28 $script:Gold $true | Out-Null
    Add-Text $slide "Critical-outage locality-hours`n$fixedOutageLabel  ->  $ppoOutageLabel" 560 400 300 54 22 $script:Ink $true 2 | Out-Null
    Add-Notes $slide ("This is the strongest proposal-facing result. The critical-service ratio is modeled critical energy served divided by critical demand. Critical-outage locality-hours use the locked 99% threshold and are diagnostic, not a bootstrap endpoint.`r`n`r`n[Sources]`r`n- docs/RESULTS.md (Reproduced paired aggregate comparison)`r`n- data/evidence/paired-benchmark.json`r`n[/Sources]")

    # Slide 8 - efficiency
    $slide = $presentation.Slides.Add(8, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 8
    Add-SlideTitle $slide "PPO serves $servedDeltaMagnitudeLabel more and uses supply more effectively"
    Add-ComparisonBar $slide "Served energy" 154 ([double]$fixed.served_mwh / [double]$fixed.available_mwh) ([double]$ppo.served_mwh / [double]$ppo.available_mwh) $fixedServedLabel $ppoServedLabel
    Add-ComparisonBar $slide "Utilization" 243 ([double]$fixed.utilization) ([double]$ppo.utilization) $fixedUtilizationLabel $ppoUtilizationLabel
    Add-ComparisonBar $slide "Curtailed energy" 332 1.0 ([double]$ppo.curtailed_mwh / [double]$fixed.curtailed_mwh) "Fixed $fixedCurtailedLabel" "PPO $ppoCurtailedLabel"
    Add-Text $slide "$servedDeltaLabel served" 70 424 270 34 24 $script:Teal $true | Out-Null
    Add-Text $slide "$utilizationDeltaLabel utilization" 357 424 270 34 24 $script:Gold $true | Out-Null
    Add-Text $slide "$curtailedDeltaLabel curtailed" 648 424 250 34 24 $script:Teal $true | Out-Null
    Add-Notes $slide ("Served and curtailed energy are complementary under the energy identity, so present them as two views of the same allocation change rather than independent endpoints.`r`n`r`n[Sources]`r`n- docs/RESULTS.md (Reproduced paired aggregate comparison)`r`n- data/evidence/paired-benchmark.json`r`n[/Sources]")

    # Slide 9 - intervals
    $slide = $presentation.Slides.Add(9, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 9 $true
    Add-SlideTitle $slide "All seven paired intervals favor PPO" $true
    Add-Text $slide "Outcome" 64 132 230 24 17 "60727D" $true | Out-Null
    Add-Text $slide "PPO - Fixed" 332 132 175 24 17 "60727D" $true | Out-Null
    Add-Text $slide "Cluster-bootstrap 95% interval" 540 132 350 24 17 "60727D" $true | Out-Null
    for ($index = 0; $index -lt $intervalRows.Count; $index++) {
        $top = 171 + ($index * 43)
        if ($index % 2 -eq 0) {
            Add-Box $slide 54 ($top - 6) 850 38 "E7ECEE" "" 1 | Out-Null
        }
        Add-Text $slide $intervalRows[$index][0] 64 $top 240 24 17 $script:Ink $(if ($index -lt 6) { $true } else { $false }) | Out-Null
        Add-Text $slide $intervalRows[$index][1] 332 $top 180 24 18 $script:Teal $true | Out-Null
        Add-Text $slide $intervalRows[$index][2] 540 $top 330 24 17 $script:Ink | Out-Null
    }
    Add-Text $slide "$($bootstrapReplicates.ToString('N0', [Globalization.CultureInfo]::InvariantCulture)) deterministic seed-stratified paired cluster replicates" 64 476 820 24 16 "60727D" | Out-Null
    Add-Notes $slide ("Positive favors PPO for higher-is-better outcomes; negative favors PPO for lower-is-better outcomes. The bootstrap resamples $independentBaseDrawCount base-draw clusters inside seed strata and keeps all $shortageVariantCount shortage variants together.`r`n`r`n[Sources]`r`n- docs/METHODOLOGY.md (Paired uncertainty calculation)`r`n- docs/RESULTS.md (Paired differences and 95% intervals)`r`n- data/evidence/paired-benchmark.json, SHA-256 $expectedEvidenceSha256`r`n[/Sources]")

    # Slide 10 - consistency
    $slide = $presentation.Slides.Add(10, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 10
    Add-SlideTitle $slide "The advantage is consistent, not driven by one seed"
    Add-Text $slide "$cellCount / $cellCount" 64 132 270 78 55 $script:Gold $true | Out-Null
    Add-Text $slide "shortage-by-seed cells favor PPO`nfor every primary outcome" 68 215 320 55 20 $script:OffWhite $true | Out-Null
    for ($index = 0; $index -lt $primaryMetricLabels.Count; $index++) {
        $top = 299 + ($index * 31)
        Add-Text $slide $primaryMetricLabels[$index] 70 $top 190 22 16 $script:OffWhite | Out-Null
        Add-Line $slide 238 ($top + 11) 405 ($top + 11) $script:Teal 5 | Out-Null
        Add-Text $slide "$($primaryFavorableCounts[$index])/$cellCount" 417 ($top - 2) 58 24 16 $script:Teal $true | Out-Null
    }
    Add-Line $slide 500 132 500 460 $script:Grid 1 | Out-Null
    Add-Text $slide "Safety and diagnostics" 548 145 330 32 24 $script:Teal $true | Out-Null
    Add-Text $slide $powerBalanceLabel 548 205 165 62 42 $script:OffWhite $true | Out-Null
    Add-Text $slide "power-balance violation rate`nin both controllers" 720 210 178 58 17 $script:OffWhite | Out-Null
    Add-Text $slide $switchingDeltaLabel 548 300 150 40 29 $script:Gold $true | Out-Null
    Add-Text $slide "aggregate switching rate" 708 307 190 26 17 $script:OffWhite | Out-Null
    Add-Text $slide "Oversubscription stays diagnostic:`ndirection was mixed across cells" 548 379 330 50 18 $script:Muted | Out-Null
    Add-Notes $slide ("All $($primaryMetricKeys.Count) primary outcomes favor PPO in all $cellCount cells. Switching favors PPO overall but in $switchingFavorableCount of $cellCount cells. Oversubscription is lower in aggregate for PPO but mixed by cell, so it is not presented as a consistent primary advantage.`r`n`r`n[Sources]`r`n- docs/RESULTS.md (Consistency across the nine evaluation cells)`r`n- data/evidence/paired-benchmark.json, SHA-256 $expectedEvidenceSha256`r`n[/Sources]")

    # Slide 11 - boundaries
    $slide = $presentation.Slides.Add(11, 12)
    Set-SlideBackground $slide $script:OffWhite
    Add-Chrome $slide 11 $true
    Add-SlideTitle $slide "The evidence answers the proposal - within a strict boundary" $true
    Add-Text $slide "What this establishes" 64 137 360 30 24 $script:Teal $true | Out-Null
    Add-BulletList $slide @(
        "PPO beats Fixed Rotation in the locked simulator",
        "Critical-service protection and fairness improve",
        "The result is reproducible from pinned artifacts",
        "The desktop demonstrates one shared scenario honestly"
    ) 64 184 390 17 $script:Ink 48 $script:Teal | Out-Null
    Add-Text $slide "What this does not establish" 522 137 370 30 24 "A15E40" $true | Out-Null
    Add-BulletList $slide @(
        "Field performance on Sudan's electrical grid",
        "Network stability, voltage, or protection behavior",
        "Superiority over every possible controller",
        "Forecast accuracy from real operational data"
    ) 522 184 390 17 $script:Ink 48 "A15E40" | Out-Null
    Add-Box $slide 64 401 828 66 "E4F4F0" $script:Teal 5 | Out-Null
    Add-Text $slide "Historical evidence stored 18 batch-cell summaries. Episode rows and intervals were regenerated from the audited simulator and frozen PPO checkpoint." 82 417 790 38 17 $script:Ink $true 2 | Out-Null
    Add-Notes $slide ("This boundary is part of the result, not a weakness to hide. The accepted historical artifact did not contain episode rows, so the paired rows and intervals are a reviewed regeneration, not reconstructed historical observations.`r`n`r`n[Sources]`r`n- docs/RESULTS.md (Two evidence layers; Limitations)`r`n- docs/METHODOLOGY.md (Limitations and threats to validity)`r`n[/Sources]")

    # Slide 12 - close
    $slide = $presentation.Slides.Add(12, 12)
    Set-SlideBackground $slide $script:Navy
    Add-Chrome $slide 12
    Add-Text $slide "THE ANSWER" 60 72 240 24 16 $script:Teal $true | Out-Null
    Add-Text $slide "PPO > Fixed Rotation" 60 128 820 70 49 $script:OffWhite $true | Out-Null
    Add-Text $slide "under the locked synthetic evaluation protocol" 62 205 660 34 24 $script:Gold | Out-Null
    Add-Line $slide 62 271 898 271 $script:Grid 1 | Out-Null
    Add-BulletList $slide @(
        "Verified - pinned model, evidence, provenance, and release gate",
        "Interactive - one shared GPU scenario with locality-level replay",
        "Honest - synthetic scope, historical boundary, and limitations visible"
    ) 62 306 810 20 $script:OffWhite 49 | Out-Null
    Add-Box $slide 62 452 836 42 $script:Teal "" 5 | Out-Null
    Add-Text $slide "Next: validate with representative operational data and electrical-network constraints" 78 463 804 22 17 $script:Navy $true 2 | Out-Null
    Add-Notes $slide ("Close by answering the original question, then transition to the live demonstration. The next scientific step is operational-data and network-constraint validation before any field-facing use.`r`n`r`n[Sources]`r`n- docs/RESULTS.md (Result statement; Interpretation)`r`n- docs/REPORT.md`r`n[/Sources]")

    $presentation.SaveAs($OutputPath, 24)

    Get-ChildItem -LiteralPath $renderDirectory -Filter "slide-*.png" -File -ErrorAction SilentlyContinue | Remove-Item -Force
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
