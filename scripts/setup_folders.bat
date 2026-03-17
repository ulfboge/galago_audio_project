@echo off
REM Set your project root folder here
set "BASE=C:\Users\galag\GitHub\galago_audio_project"

cd /d "%BASE%" || (
    echo Could not change directory to %BASE%
    pause
    exit /b 1
)

REM Create main folders (if they don't already exist)
mkdir "audio_clips" 2>nul
mkdir "melspectrograms" 2>nul

REM List of canonical species folder names
for %%S in (
 "Otolemur_crassicaudatus"
 "Otolemur_garnettii"
 "Galago_senegalensis"
 "Galago_moholi"
 "Galago_gallarum"
 "Galago_matschiei"
 "Paragalago_granti"
 "Paragalago_zanzibaricus"
 "Paragalago_cocos"
 "Paragalago_rondoensis"
 "Paragalago_orinus"
 "Paragalago_arthuri"
 "Galagoides_demidovii"
 "Galagoides_thomasi"
 "Galagoides_kumbakumba"
 "Galagoides_phasma"
 "Galagoides_sp_nov"
 "Galagoides_kumbirensis"
 "Sciurocheirus_gabonensis"
 "Sciurocheirus_alleni"
 "Sciurocheirus_cameronensis"
 "Sciurocheirus_makandensis"
 "Euoticus_elegantulus"
 "Euoticus_pallidus"
) do (
    mkdir "audio_clips\%%~S" 2>nul
    mkdir "melspectrograms\%%~S" 2>nul
)

echo Done creating species folders.
pause
