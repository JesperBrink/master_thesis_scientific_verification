import Autocomplete from '@material-ui/lab/Autocomplete';
import TextField from '@material-ui/core/TextField';
import Grid from '@material-ui/core/Grid';

export default function Filters({ hyperparams, onChange }) {
    const getFilterDropdowns = () => {
        let filters = [];

        for (const [key, values] of Object.entries(hyperparams)) {
            filters.push(
                <Grid item xs={4} key={key}>
                    <Autocomplete
                        id={`filter_${key}`}
                        options={Array.from(values)}
                        onChange={(event, value, reason) => onChange(value, key)}
                        multiple
                        getOptionLabel={(option) => option.toString()}
                        renderInput={(params) => <TextField {...params} label={key} variant="outlined"/>}
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