# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Docker launch script for Alphafold docker image."""

import os
import pathlib
import signal
from typing import Tuple
import shutil
import subprocess

from absl import app
from absl import flags
from absl import logging
import docker
from docker import types


flags.DEFINE_bool(
    'use_gpu', True, 'Enable NVIDIA runtime to run with GPUs.')
flags.DEFINE_string(
    'gpu_devices', 'all',
    'Comma separated list of devices to pass to NVIDIA_VISIBLE_DEVICES.')
flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')
flags.DEFINE_list(
    'is_prokaryote_list', None, 'Optional for multimer system, not used by the '
    'single chain system. This list should contain a boolean for each fasta '
    'specifying true where the target complex is from a prokaryote, and false '
    'where it is not, or where the origin is unknown. These values determine '
    'the pairing method for the MSA.')
flags.DEFINE_string(
    'output_dir', '/tmp/alphafold',
    'Path to a directory that will store the results.')
flags.DEFINE_string(
    'data_dir', None,
    'Path to directory with supporting data: AlphaFold parameters and genetic '
    'and template databases. Set to the target of download_all_databases.sh.'
    'If use_templates is False and use_precomputed_msas is True, and all MSAs '
    'have indeed been precomputed, only the AlphaFold parameters'
    '(not the databases) are required.')
flags.DEFINE_string(
    'docker_image_name', 'alphafold', 'Name of the AlphaFold Docker image.')
flags.DEFINE_string(
    'max_template_date', None,
    'Maximum template release date to consider (ISO-8601 format: YYYY-MM-DD). '
    'Important if folding historical test sets.')
flags.DEFINE_enum(
    'db_preset', 'full_dbs', ['full_dbs', 'reduced_dbs'],
    'Choose preset MSA database configuration - smaller genetic database '
    'config (reduced_dbs) or full genetic database config (full_dbs)')
flags.DEFINE_enum(
    'model_preset', 'monomer',
    ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
    'Choose preset model configuration - the monomer model, the monomer model '
    'with extra ensembling, monomer model with pTM head, or multimer model')
flags.DEFINE_boolean(
    'benchmark', False,
    'Run multiple JAX model evaluations to obtain a timing that excludes the '
    'compilation time, which should be more indicative of the time required '
    'for inferencing many proteins.')
flags.DEFINE_integer(
    'random_seed', None, 'The random seed for the data '
    'pipeline. By default, this is randomly generated. Note '
    'that even if this is set, Alphafold may still not be '
    'deterministic, because processes like GPU inference are '
    'nondeterministic.')
flags.DEFINE_boolean(
    'use_precomputed_msas', False,
    'Whether to read MSAs that have been written to disk. WARNING: This will '
    'not check if the sequence, database or configuration have changed.')
flags.DEFINE_boolean(
    'only_msas', False, 
    'Whether to only build MSAs, and not do any prediction.')
flags.DEFINE_boolean(
    'amber', True,
    'Whether to do an Amber relaxation of the models.')
flags.DEFINE_boolean(
    'use_templates', True,
    'Whether to search for template structures.')
flags.DEFINE_boolean(
    'singularity', False,
    'Whether to use Singularity instead of Docker to execute AlphaFold. '
    'In that case, $SINGULARITY_IMAGE_DIR must be defined and '
    'must contain alphafold.simg or alphafold.sif')
flags.DEFINE_boolean(
    'dev', False, 'Run inside alphafold-dev Docker container. '
    'This is meant for modifying AlphaFold without re-building the Docker image '
    'The AlphaFold source code is bound to the container '
    'from an external location, defined as environment variable ALPHAFOLD_DIR.'
    'ALPHAFOLD_DIR should be set to <AlphaFold git repo dir>/alphafold.'
)

FLAGS = flags.FLAGS

_ROOT_MOUNT_DIRECTORY = '/mnt/'


def _create_mount(mount_name: str, path: str) -> Tuple[types.Mount, str]:
  path = os.path.abspath(path)
  source_path = os.path.dirname(path)
  target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, mount_name)
  logging.info('Mounting %s -> %s', source_path, target_path)
  mount = types.Mount(target_path, source_path, type='bind', read_only=True)
  return mount, os.path.join(target_path, os.path.basename(path))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.use_templates:
    if FLAGS.max_template_date is None:
      raise app.UsageError(
        'If templates are used, max_template_date must be defined')

  require_all_databases = True
  if FLAGS.use_precomputed_msas and not FLAGS.use_templates \
      and not FLAGS.only_msas:
    require_all_databases = False

  # You can individually override the following paths if you have placed the
  # data in locations other than the FLAGS.data_dir.

  # Path to the Uniref90 database for use by JackHMMER.
  uniref90_database_path = os.path.join(
      FLAGS.data_dir, 'uniref90', 'uniref90.fasta')

  # Path to the Uniprot database for use by JackHMMER.
  uniprot_database_path = os.path.join(
      FLAGS.data_dir, 'uniprot', 'uniprot.fasta')

  # Path to the MGnify database for use by JackHMMER.
  mgnify_database_path = os.path.join(
      FLAGS.data_dir, 'mgnify', 'mgy_clusters_2018_12.fa')

  # Path to the BFD database for use by HHblits.
  bfd_database_path = os.path.join(
      FLAGS.data_dir, 'bfd',
      'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')

  # Path to the Small BFD database for use by JackHMMER.
  small_bfd_database_path = os.path.join(
      FLAGS.data_dir, 'small_bfd', 'bfd-first_non_consensus_sequences.fasta')

  # Path to the Uniclust30 database for use by HHblits.
  uniclust30_database_path = os.path.join(
      FLAGS.data_dir, 'uniclust30', 'uniclust30_2018_08', 'uniclust30_2018_08')

  # Path to the PDB70 database for use by HHsearch.
  pdb70_database_path = os.path.join(FLAGS.data_dir, 'pdb70', 'pdb70')

  # Path to the PDB seqres database for use by hmmsearch.
  pdb_seqres_database_path = os.path.join(
      FLAGS.data_dir, 'pdb_seqres', 'pdb_seqres.txt')

  # Path to a directory with template mmCIF structures, each named <pdb_id>.cif.
  template_mmcif_dir = os.path.join(FLAGS.data_dir, 'pdb_mmcif', 'mmcif_files')

  # Path to a file mapping obsolete PDB IDs to their replacements.
  obsolete_pdbs_path = os.path.join(FLAGS.data_dir, 'pdb_mmcif', 'obsolete.dat')

  if not FLAGS.dev:
    alphafold_path = pathlib.Path(__file__).parent.parent
    data_dir_path = pathlib.Path(FLAGS.data_dir)
    if alphafold_path == data_dir_path or alphafold_path in data_dir_path.parents:
        raise app.UsageError(
            f'The download directory {FLAGS.data_dir} should not be a subdirectory '
            f'in the AlphaFold repository directory. If it is, the Docker build is '
            f'slow since the large databases are copied during the image creation.')

  mounts = []
  command_args = []

  # Mount each fasta path as a unique target directory.
  target_fasta_paths = []
  for i, fasta_path in enumerate(FLAGS.fasta_paths):
    mount, target_path = _create_mount(f'fasta_path_{i}', fasta_path)
    mounts.append(mount)
    target_fasta_paths.append(target_path)
  command_args.append(f'--fasta_paths={",".join(target_fasta_paths)}')

  database_paths = [
      ('uniref90_database_path', uniref90_database_path),
      ('mgnify_database_path', mgnify_database_path),
      ('data_dir', FLAGS.data_dir),
      ('template_mmcif_dir', template_mmcif_dir),
      ('obsolete_pdbs_path', obsolete_pdbs_path),
  ]

  if FLAGS.model_preset == 'multimer':
    database_paths.append(('uniprot_database_path', uniprot_database_path))
    database_paths.append(('pdb_seqres_database_path',
                           pdb_seqres_database_path))
  else:
    database_paths.append(('pdb70_database_path', pdb70_database_path))

  if FLAGS.db_preset == 'reduced_dbs':
    database_paths.append(('small_bfd_database_path', small_bfd_database_path))
  else:
    database_paths.extend([
        ('uniclust30_database_path', uniclust30_database_path),
        ('bfd_database_path', bfd_database_path),
    ])
  if not require_all_databases:
    database_paths2 = []
    for db_name, db_path in database_paths:
      if db_name == "data_dir" or os.path.exists(db_path) \
          or os.path.exists(db_path + "_hhm.ffindex"):
        database_paths2.append((db_name, db_path))
    database_paths = database_paths2
  for name, path in database_paths:
    if path:
      mount, target_path = _create_mount(name, path)
      mounts.append(mount)
      command_args.append(f'--{name}={target_path}')

  output_target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, 'output')
  output_dir = os.path.abspath(FLAGS.output_dir)
  mounts.append(types.Mount(output_target_path, output_dir, type='bind'))

  command_args.extend([
      f'--output_dir={output_target_path}',
      f'--max_template_date={FLAGS.max_template_date}',
      f'--db_preset={FLAGS.db_preset}',
      f'--model_preset={FLAGS.model_preset}',
      f'--benchmark={FLAGS.benchmark}',
      f'--use_precomputed_msas={FLAGS.use_precomputed_msas}',
      f'--only_msas={FLAGS.only_msas}',
      f'--amber={FLAGS.amber}',
      f'--use_templates={FLAGS.use_templates}',
      '--logtostderr',
  ])
  if FLAGS.random_seed is not None:
      command_args.append(
        f'--random_seed={FLAGS.random_seed}')

  if FLAGS.is_prokaryote_list:
    command_args.append(
        f'--is_prokaryote_list={",".join(FLAGS.is_prokaryote_list)}')

  docker_image_name = FLAGS.docker_image_name
  if FLAGS.dev: 
    alphafold_dir = os.environ.get("ALPHAFOLD_DIR")
    if alphafold_dir is None:
      raise app.UsageError('ALPHAFOLD_DIR is undefined')
    run_alphafold_py = os.path.join(alphafold_dir, "run_alphafold.py")
    if not os.path.exists(run_alphafold_py):
      raise app.UsageError('ALPHAFOLD_DIR must contain "run_alphafold.py"')
    mounts.append(types.Mount(
      "/app/alphafold", alphafold_dir, 
      type='bind', read_only=True)
    )
    if docker_image_name == "alphafold":
      docker_image_name = "alphafold-dev"

  if FLAGS.singularity:
    if shutil.which("singularity") is None:
      raise app.UsageError('Could not find path to the "singularity" binary. '
                           'Make sure it is installed on your system.')
    singularity_image_dir = os.environ.get("SINGULARITY_IMAGE_DIR")
    if singularity_image_dir is None:
      raise app.UsageError('SINGULARITY_IMAGE_DIR must be defined')
    if not os.path.exists(singularity_image_dir):
      raise app.UsageError(f'SINGULARITY_IMAGE_DIR {singularity_image_dir} does not exist')
    singularity_image_head = os.path.join(singularity_image_dir, FLAGS.docker_image_name)
    for singularity_ext in "sif", "simg":
      singularity_image =  singularity_image_head + "." + singularity_ext
      if os.path.exists(singularity_image):
        break
    else:
      raise app.UsageError(f'SINGULARITY_IMAGE_DIR {singularity_image_dir} does not contain '
      f'{FLAGS.docker_image_name}.sif/.simg')
    singularity_command = ["singularity", "run", "--cleanenv"]
    singularity_command += ["--env", "TF_FORCE_UNIFIED_MEMORY=1"]
    singularity_command += ["--env", "XLA_PYTHON_CLIENT_MEM_FRACTION=4.0"]
    singularity_command += ["--env", "OPENMM_CPU_THREADS=8"]      
    if FLAGS.use_gpu:
      singularity_command += ["--nv"]
      if FLAGS.gpu_devices != "all":
        os.environ["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_devices
    for mount in mounts:
      mount_readonly = "ro" if mount["ReadOnly"] else "rw"
      singularity_command += ['--bind', f'{mount["Source"]}:{mount["Target"]}:{mount_readonly}']
    singularity_command += [singularity_image]
    singularity_command += command_args
    print(" ".join(singularity_command))
    subprocess.run(singularity_command)
  else:   # execute with Docker
    client = docker.from_env()
    container = client.containers.run(
        image=docker_image_name,
        command=command_args,
        runtime='nvidia' if FLAGS.use_gpu else None,
        remove=True,
        detach=True,
        mounts=mounts,
        environment={
            'NVIDIA_VISIBLE_DEVICES': FLAGS.gpu_devices,
            # The following flags allow us to make predictions on proteins that
            # would typically be too long to fit into GPU memory.
            'TF_FORCE_UNIFIED_MEMORY': '1',
            'XLA_PYTHON_CLIENT_MEM_FRACTION': '4.0',
        })

    # Add signal handler to ensure CTRL+C also stops the running container.
    signal.signal(signal.SIGINT,
                  lambda unused_sig, unused_frame: container.kill())

    for line in container.logs(stream=True):
      logging.info(line.strip().decode('utf-8'))


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'data_dir',
      'fasta_paths'
  ])
  app.run(main)
