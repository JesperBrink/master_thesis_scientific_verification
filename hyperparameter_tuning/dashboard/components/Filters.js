import Autocomplete from '@material-ui/lab/Autocomplete';
import TextField from '@material-ui/core/TextField';
import Grid from '@material-ui/core/Grid';

export default function Filters({ hyperparams, onChange, filters }) {
    const getFilterDropdowns = () => {
        let filterDropdowns = [];

        for (const [key, values] of Object.entries(hyperparams)) {
            filterDropdowns.push(
                <Grid item xs={4} key={key}>
                    <Autocomplete
                        id={`filter_${key}`}
                        options={Array.from(values)}
                        onChange={(event, value, reason) => onChange(value, key)}
                        multiple
                        value={Array.from(values).filter(e => filters[key] ? filters[key].includes(e) : false)}
                        getOptionLabel={(option) => option.toString()}
                        renderInput={(params) => <TextField {...params} label={key} variant="outlined"/>}
                    />
                </Grid>
            )
        }

        return filterDropdowns;
    }

    return (
        getFilterDropdowns()
    )
}