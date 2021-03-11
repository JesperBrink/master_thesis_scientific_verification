import Grid from '@material-ui/core/Grid';
import { makeStyles } from '@material-ui/core/styles';
import UploadButton from '../components/UploadButton';

export default function Home() {
    // TODO: Checlist:
        // row med upload button
        // parse data to list of objects
        // name, class weight 1, class weight 2, #units, #scifact_epochs, #fever_epochs, threshold, results: map with the various results
        // if data, show row with table (datagrid) 
            // pages i table?
            // Standard sort efter f1
        // row with multi-dropdowns to filter data (skal egentlig være over table)
        // lav dem baseret på indlæst data? I.e. tag værdierne fra daten i stedet for hardcoded. Lad være med at vise dropdown hvis der ikke er nogle værdier (løser problemet med at stance-selection har færre hyperparameters)
        // select rows (in table) to compare in another table (which is only shown if at least two datapoints are selected)

    return (
        <div >
            <Grid 
                container 
                spacing={2}
                direction="column"
                alignContent="center"
                style={{ flexGrow: 1, paddingTop: 30 }}
            >
                <Grid item xs>
                    {"Upload button"}
                    {/* <UploadButton readResultsFile={readResultsFile}/> */}
                </Grid>
                <Grid item xs>
                    {"Dropdowns"}
                </Grid>
                <Grid item xs>
                    {"Table"}
                </Grid>
            </Grid>
        </div>
    )
}
