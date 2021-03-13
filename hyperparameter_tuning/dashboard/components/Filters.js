import Autocomplete from '@material-ui/lab/Autocomplete';
import TextField from '@material-ui/core/TextField';
import Grid from '@material-ui/core/Grid';

export default function Filters({ hyperparams }) {
    const getFilterDropdowns = () => {
        let filters = [];

        for (const [key, values] of Object.entries(hyperparams)) {
            console.log(key, values)
            filters.push(
                <Grid item xs={4} key={key}>
                    <Autocomplete
                        id={`filter_${key}`}
                        options={Array.from(values)}
                        multiple
                        getOptionLabel={(option) => option.toString()}
                        renderInput={(params) => <TextField {...params} label={key} />}
                    />
                </Grid>
            )
        }

        return filters;
    }

    return (
        getFilterDropdowns()
    )
}